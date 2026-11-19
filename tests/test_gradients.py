import torch
from blockvga import BlockGridConfig, BlockGridLayer

def test_gradients_flow():
    """
    test_gradients: torch.autograd.gradcheck (float64) with gamma = 1.0; gradients
    flow through tied w_a into BOTH pool and readback, into the routing-feature
    projection, and into A_k, B_k when inner_op is on.
    """
    H, W, P, Q, F_dim, R, K, d = 2, 2, 2, 2, 2, 2, 4, 8
    cfg = BlockGridConfig(
        H=H, W=W, P=P, Q=Q, F=F_dim, R=R, K=K, d=d,
        inner_op=True, gamma_init=1.0, tau_init=1.0
    )
    layer = BlockGridLayer(cfg).to(torch.float64)
    
    X = torch.randn(1, H, W, P, Q, F_dim, dtype=torch.float64, requires_grad=True)
    rf = torch.randn(1, H, W, R, dtype=torch.float64, requires_grad=True)
    
    out = layer(X, routing_features=rf)
    loss = out.sum()
    loss.backward()
    
    # Check gradients in inputs
    assert X.grad is not None and X.grad.norm().item() > 0
    assert rf.grad is not None and rf.grad.norm().item() > 0
    
    # Check gradients in routing projections
    assert layer.assignment.linear_r.weight.grad is not None
    assert layer.assignment.linear_r.weight.grad.norm().item() > 0
    assert layer.assignment.W_c.weight.grad is not None
    assert layer.assignment.W_c.weight.grad.norm().item() > 0
    assert layer.assignment.W_p.weight.grad is not None
    assert layer.assignment.W_p.weight.grad.norm().item() > 0
    
    # Check gradients in slot mixing and inner operators
    assert layer.slotmix.G.grad is not None and layer.slotmix.G.grad.norm().item() > 0
    assert layer.inner.A.grad is not None and layer.inner.A.grad.norm().item() > 0
    assert layer.inner.B.grad is not None and layer.inner.B.grad.norm().item() > 0
    assert layer.proj.weight.grad is not None and layer.proj.weight.grad.norm().item() > 0

def test_gradcheck_analytical():
    # Mini numerical gradcheck with float64
    H, W, P, Q, F_dim, R, K, d = 2, 2, 2, 2, 2, 0, 3, 4
    cfg = BlockGridConfig(
        H=H, W=W, P=P, Q=Q, F=F_dim, R=R, K=K, d=d,
        inner_op=True, gamma_init=1.0, tau_init=1.0
    )
    layer = BlockGridLayer(cfg).to(torch.float64)
    X = torch.randn(1, H, W, P, Q, F_dim, dtype=torch.float64, requires_grad=True)
    
    def forward_fn(x):
        return layer(x)
        
    assert torch.autograd.gradcheck(forward_fn, (X,), eps=1e-6, atol=1e-4, rtol=1e-3)
