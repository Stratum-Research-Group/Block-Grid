import torch
import triton
import triton.language as tl

@triton.jit
def _fused_pool_fwd_kernel(
    V_ptr, W_ptr, Z_ptr,
    B, N, K, C,
    stride_vb, stride_vn, stride_vc,
    stride_wb, stride_wn, stride_wk,
    stride_zb, stride_zk, stride_zc,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        w_ptrs = W_ptr + pid_b * stride_wb + offs_n * stride_wn + pid_k * stride_wk
        w_val = tl.load(w_ptrs, mask=mask_n, other=0.0)

        v_ptrs = V_ptr + pid_b * stride_vb + offs_n[:, None] * stride_vn + offs_c[None, :] * stride_vc
        v_val = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_c[None, :]), other=0.0)

        acc += tl.sum(w_val[:, None] * v_val, axis=0)

    z_ptrs = Z_ptr + pid_b * stride_zb + pid_k * stride_zk + offs_c * stride_zc
    tl.store(z_ptrs, acc, mask=mask_c)

@triton.jit
def _fused_pool_bwd_dv_kernel(
    dZ_ptr, W_ptr, dV_ptr,
    B, N, K, C,
    stride_dzb, stride_dzk, stride_dzc,
    stride_wb, stride_wn, stride_wk,
    stride_dvb, stride_dvn, stride_dvc,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    """
    Computes dV[b, n, c] = sum_{k} W[b, n, k] * dZ[b, k, c]
    Triton program grid: (B, N, c_tiles)
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        w_ptrs = W_ptr + pid_b * stride_wb + pid_n * stride_wn + offs_k * stride_wk
        w_val = tl.load(w_ptrs, mask=mask_k, other=0.0) # [BLOCK_K]

        dz_ptrs = dZ_ptr + pid_b * stride_dzb + offs_k[:, None] * stride_dzk + offs_c[None, :] * stride_dzc
        dz_val = tl.load(dz_ptrs, mask=(mask_k[:, None] & mask_c[None, :]), other=0.0) # [BLOCK_K, BLOCK_C]

        acc += tl.sum(w_val[:, None] * dz_val, axis=0)

    dv_ptrs = dV_ptr + pid_b * stride_dvb + pid_n * stride_dvn + offs_c * stride_dvc
    tl.store(dv_ptrs, acc, mask=mask_c)

@triton.jit
def _fused_pool_bwd_dw_kernel(
    dZ_ptr, V_ptr, dW_ptr,
    B, N, K, C,
    stride_dzb, stride_dzk, stride_dzc,
    stride_vb, stride_vn, stride_vc,
    stride_dwb, stride_dwn, stride_dwk,
    BLOCK_C: tl.constexpr
):
    """
    Computes dW[b, n, k] = sum_{c} V[b, n, c] * dZ[b, k, c]
    Triton program grid: (B, N, K)
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    acc = 0.0
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        v_ptrs = V_ptr + pid_b * stride_vb + pid_n * stride_vn + offs_c * stride_vc
        v_val = tl.load(v_ptrs, mask=mask_c, other=0.0)

        dz_ptrs = dZ_ptr + pid_b * stride_dzb + pid_k * stride_dzk + offs_c * stride_dzc
        dz_val = tl.load(dz_ptrs, mask=mask_c, other=0.0)

        acc += tl.sum(v_val * dz_val)

    dw_ptr = dW_ptr + pid_b * stride_dwb + pid_n * stride_dwn + pid_k * stride_dwk
    tl.store(dw_ptr, acc)

class FusedBlockPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        V: [B, N, C]
        W: [B, N, K]
        Returns:
            Z: [B, K, C]
        """
        B, N, C = V.shape
        _, _, K = W.shape

        V_contig = V.contiguous()
        W_contig = W.contiguous()
        Z = torch.empty((B, K, C), device=V.device, dtype=torch.float32)

        BLOCK_N = 64
        BLOCK_C = 32
        grid = (B, K, triton.cdiv(C, BLOCK_C))

        _fused_pool_fwd_kernel[grid](
            V_contig, W_contig, Z,
            B, N, K, C,
            V_contig.stride(0), V_contig.stride(1), V_contig.stride(2),
            W_contig.stride(0), W_contig.stride(1), W_contig.stride(2),
            Z.stride(0), Z.stride(1), Z.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C
        )

        ctx.save_for_backward(V_contig, W_contig)
        ctx.shape_info = (B, N, K, C)
        return Z.to(V.dtype)

    @staticmethod
    def backward(ctx, dZ: torch.Tensor):
        V_contig, W_contig = ctx.saved_tensors
        B, N, K, C = ctx.shape_info
        dZ_contig = dZ.contiguous().float()

        # Compute dV: [B, N, C]
        dV = torch.empty((B, N, C), device=V_contig.device, dtype=torch.float32)
        BLOCK_K = min(32, K)
        BLOCK_C = 32
        grid_dv = (B, N, triton.cdiv(C, BLOCK_C))

        _fused_pool_bwd_dv_kernel[grid_dv](
            dZ_contig, W_contig, dV,
            B, N, K, C,
            dZ_contig.stride(0), dZ_contig.stride(1), dZ_contig.stride(2),
            W_contig.stride(0), W_contig.stride(1), W_contig.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_C=BLOCK_C
        )

        # Compute dW: [B, N, K]
        dW = torch.empty((B, N, K), device=W_contig.device, dtype=torch.float32)
        grid_dw = (B, N, K)

        _fused_pool_bwd_dw_kernel[grid_dw](
            dZ_contig, V_contig, dW,
            B, N, K, C,
            dZ_contig.stride(0), dZ_contig.stride(1), dZ_contig.stride(2),
            V_contig.stride(0), V_contig.stride(1), V_contig.stride(2),
            dW.stride(0), dW.stride(1), dW.stride(2),
            BLOCK_C=BLOCK_C
        )

        return dV.to(V_contig.dtype), dW.to(W_contig.dtype)

def triton_fused_autograd_pool(V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    End-to-end autograd differentiable fused pooling operator.
    V: [..., C]
    W: [..., K]
    """
    orig_shape_v = V.shape
    C = orig_shape_v[-1]
    K = W.shape[-1]
    B = orig_shape_v[0]
    N = V.numel() // (B * C)

    V_flat = V.view(B, N, C)
    W_flat = W.view(B, N, K)
    Z = FusedBlockPoolFunction.apply(V_flat, W_flat)
    return Z
