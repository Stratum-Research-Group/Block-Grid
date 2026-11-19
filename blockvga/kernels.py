import torch
import triton
import triton.language as tl

@triton.jit
def _fused_pool_kernel(
    V_ptr,        # [B, N, C]
    W_ptr,        # [B, N, K]
    Z_ptr,        # [B, K, C]
    B, N, K, C,
    stride_vb, stride_vn, stride_vc,
    stride_wb, stride_wn, stride_wk,
    stride_zb, stride_zk, stride_zc,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    """
    Computes Z[b, k, c] = sum_{n} W[b, n, k] * V[b, n, c]
    Triton program grid: (B, K, c_tiles)
    Each program instance computes a tile of size (1, 1, BLOCK_C) for a specific (b, k).
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Offsets in channel dimension C
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Accumulator for Z[b, k, c]
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over N in tiles of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Load W tile: [BLOCK_N] for current (b, n, k)
        w_ptrs = W_ptr + pid_b * stride_wb + offs_n * stride_wn + pid_k * stride_wk
        w_val = tl.load(w_ptrs, mask=mask_n, other=0.0) # [BLOCK_N]

        # Load V tile: [BLOCK_N, BLOCK_C] for current (b, n, c)
        v_ptrs = V_ptr + pid_b * stride_vb + offs_n[:, None] * stride_vn + offs_c[None, :] * stride_vc
        v_val = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_c[None, :]), other=0.0) # [BLOCK_N, BLOCK_C]

        # Multiply and accumulate along N: w_val [BLOCK_N, 1] * v_val [BLOCK_N, BLOCK_C]
        acc += tl.sum(w_val[:, None] * v_val, axis=0)

    # Store result to Z[b, k, c]
    z_ptrs = Z_ptr + pid_b * stride_zb + pid_k * stride_zk + offs_c * stride_zc
    tl.store(z_ptrs, acc, mask=mask_c)

@triton.jit
def _fused_readback_kernel(
    Z_ptr,        # [B, K, C]
    R_ptr,        # [B, N, K]
    Y_ptr,        # [B, N, C]
    B, N, K, C,
    stride_zb, stride_zk, stride_zc,
    stride_rb, stride_rn, stride_rk,
    stride_yb, stride_yn, stride_yc,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    """
    Computes Y[b, n, c] = sum_{k} R[b, n, k] * Z[b, k, c]
    Triton program grid: (B, N, c_tiles)
    Each program instance computes a tile of size (1, 1, BLOCK_C) for a specific (b, n).
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Offsets in channel dimension C
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Accumulator for Y[b, n, c]
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over K in tiles of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load R tile: [BLOCK_K] for current (b, n, k)
        r_ptrs = R_ptr + pid_b * stride_rb + pid_n * stride_rn + offs_k * stride_rk
        r_val = tl.load(r_ptrs, mask=mask_k, other=0.0) # [BLOCK_K]

        # Load Z tile: [BLOCK_K, BLOCK_C] for current (b, k, c)
        z_ptrs = Z_ptr + pid_b * stride_zb + offs_k[:, None] * stride_zk + offs_c[None, :] * stride_zc
        z_val = tl.load(z_ptrs, mask=(mask_k[:, None] & mask_c[None, :]), other=0.0) # [BLOCK_K, BLOCK_C]

        # Multiply and accumulate along K: r_val [BLOCK_K, 1] * z_val [BLOCK_K, BLOCK_C]
        acc += tl.sum(r_val[:, None] * z_val, axis=0)

    # Store result to Y[b, n, c]
    y_ptrs = Y_ptr + pid_b * stride_yb + pid_n * stride_yn + offs_c * stride_yc
    tl.store(y_ptrs, acc, mask=mask_c)

def triton_pool(V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    V: [B, H, W, P, Q, F] -> flattened to [B, N, C]
    W: [B, H, W, K]       -> flattened to [B, N, K]
    Returns:
        Z: [B, K, P, Q, F]
    """
    B, H, W_grid, P, Q, F_dim = V.shape
    N = H * W_grid
    C = P * Q * F_dim
    K = W.shape[-1]

    V_flat = V.contiguous().view(B, N, C)
    W_flat = W.contiguous().view(B, N, K)
    Z = torch.empty(B, K, C, device=V.device, dtype=V.dtype)

    BLOCK_N = 64
    BLOCK_C = 32
    grid = (B, K, triton.cdiv(C, BLOCK_C))

    _fused_pool_kernel[grid](
        V_flat, W_flat, Z,
        B, N, K, C,
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        W_flat.stride(0), W_flat.stride(1), W_flat.stride(2),
        Z.stride(0), Z.stride(1), Z.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C
    )

    return Z.view(B, K, P, Q, F_dim)

def triton_readback(Z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Z: [B, K, P, Q, F] -> flattened to [B, K, C]
    R: [B, H, W, K]    -> flattened to [B, N, K]
    Returns:
        Y: [B, H, W, P, Q, F]
    """
    B, K, P, Q, F_dim = Z.shape
    B_r, H, W_grid, K_r = R.shape
    N = H * W_grid
    C = P * Q * F_dim

    Z_flat = Z.contiguous().view(B, K, C)
    R_flat = R.contiguous().view(B, N, K)
    Y = torch.empty(B, N, C, device=Z.device, dtype=Z.dtype)

    BLOCK_K = min(32, K)
    BLOCK_C = 32
    grid = (B, N, triton.cdiv(C, BLOCK_C))

    _fused_readback_kernel[grid](
        Z_flat, R_flat, Y,
        B, N, K, C,
        Z_flat.stride(0), Z_flat.stride(1), Z_flat.stride(2),
        R_flat.stride(0), R_flat.stride(1), R_flat.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_K=BLOCK_K, BLOCK_C=BLOCK_C
    )

    return Y.view(B, H, W_grid, P, Q, F_dim)
