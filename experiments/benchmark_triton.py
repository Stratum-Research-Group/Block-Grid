import time
import torch
from blockgrid.pooling import BlockPooling
from blockgrid.kernels import triton_pool, triton_readback

def benchmark_triton_speed():
    device = "cuda"
    B = 4
    H = 64
    W = 64
    N = H * W # 4096 cells
    P = 4
    Q = 4
    F_dim = 4 # C = 64
    K = 32

    print(f"=== Triton Kernel Benchmark (Grid {H}x{W} = {N} cells, K={K}, C={P*Q*F_dim}) ===")
    V = torch.randn(B, H, W, P, Q, F_dim, device=device, dtype=torch.float32)
    w = torch.softmax(torch.randn(B, H, W, K, device=device, dtype=torch.float32), dim=-1)
    Z = torch.randn(B, K, P, Q, F_dim, device=device, dtype=torch.float32)

    pooling = BlockPooling(normalize_slots=False)

    # Warmup
    for _ in range(20):
        _ = pooling.pool(V, w)
        _ = triton_pool(V, w)
        _ = pooling.readback(Z, w)
        _ = triton_readback(Z, w)
    torch.cuda.synchronize()

    # 1. Benchmark Pool
    reps = 200
    t0 = time.time()
    for _ in range(reps):
        _ = pooling.pool(V, w)
    torch.cuda.synchronize()
    t_einsum_pool = (time.time() - t0) / reps * 1000

    t0 = time.time()
    for _ in range(reps):
        _ = triton_pool(V, w)
    torch.cuda.synchronize()
    t_triton_pool = (time.time() - t0) / reps * 1000

    # 2. Benchmark Readback
    t0 = time.time()
    for _ in range(reps):
        _ = pooling.readback(Z, w)
    torch.cuda.synchronize()
    t_einsum_read = (time.time() - t0) / reps * 1000

    t0 = time.time()
    for _ in range(reps):
        _ = triton_readback(Z, w)
    torch.cuda.synchronize()
    t_triton_read = (time.time() - t0) / reps * 1000

    print(f"  Pool (Einsum):    {t_einsum_pool:.4f} ms | Pool (Triton):    {t_triton_pool:.4f} ms | Speedup: {t_einsum_pool/t_triton_pool:.2f}x")
    print(f"  Readback (Einsum):{t_einsum_read:.4f} ms | Readback (Triton):{t_triton_read:.4f} ms | Speedup: {t_einsum_read/t_triton_read:.2f}x")

if __name__ == "__main__":
    benchmark_triton_speed()
