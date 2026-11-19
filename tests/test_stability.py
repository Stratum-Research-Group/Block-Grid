import torch
from blockgrid import BlockGridConfig, BlockGridLayer

def test_stability_tau_floor():
    """
    test_stability: 100 random forward/backward passes at the tau floor, K = 128;
    no NaN/Inf; normalize_slots=True with a zero-mass slot yields 0, not NaN.
    """
    H, W, P, Q, F_dim, K, d = 4, 4, 2, 2, 4, 128, 32
    cfg = BlockGridConfig(
        H=H, W=W, P=P, Q=Q, F=F_dim, K=K, d=d,
        tau_init=0.01,  # Below 0.05 floor, will be clamped
        normalize_slots=True,
        inner_op=True,
        gamma_init=0.1
    )
    layer = BlockGridLayer(cfg)
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    
    for i in range(100):
        optimizer.zero_grad()
        X = torch.randn(2, H, W, P, Q, F_dim, requires_grad=True)
        out = layer(X)
        assert not torch.isnan(out).any(), f"NaN in output at step {i}"
        assert not torch.isinf(out).any(), f"Inf in output at step {i}"
        
        loss = out.sum()
        loss.backward()
        
        assert not torch.isnan(X.grad).any(), f"NaN in gradient at step {i}"
        assert not torch.isinf(X.grad).any(), f"Inf in gradient at step {i}"
        optimizer.step()

def test_zero_mass_slot_stability():
    """
    normalize_slots=True with a zero-mass slot yields 0 via eps, never NaN.
    """
    from blockgrid.pooling import BlockPooling
    pooling = BlockPooling(normalize_slots=True, eps=1e-6)
    
    B, H, W, P, Q, F_dim, K = 2, 4, 4, 2, 2, 4, 8
    V = torch.randn(B, H, W, P, Q, F_dim)
    w = torch.zeros(B, H, W, K)
    # Slot 0 has mass, slots 1-7 have exact 0 mass
    w[..., 0] = 1.0
    
    Z = pooling.pool(V, w)
    assert not torch.isnan(Z).any()
    assert not torch.isinf(Z).any()
    # For slots 1-7, Z should be exactly 0
    assert (Z[:, 1:] == 0.0).all()
