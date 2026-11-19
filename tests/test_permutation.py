import torch
from blockvga import BlockGridConfig, BlockGridLayer

def test_permutation_equivariance():
    """
    test_permutation: permuting cells together with their positions and routing
    features permutes output blocks identically (tolerance 1e-5).
    """
    torch.manual_seed(42)
    B, H, W, P, Q, F_dim, R = 2, 4, 4, 2, 2, 4, 2
    cfg = BlockGridConfig(H=H, W=W, P=P, Q=Q, F=F_dim, R=R, K=8, d=16, inner_op=True)
    layer = BlockGridLayer(cfg)
    layer.eval()

    X = torch.randn(B, H, W, P, Q, F_dim)
    rf = torch.randn(B, H, W, R)
    
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    positions = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    # Original forward
    out_orig = layer(X, routing_features=rf, positions=positions)
    
    # Flatten spatial grid H*W = 16
    N = H * W
    X_flat = X.view(B, N, P, Q, F_dim)
    rf_flat = rf.view(B, N, R)
    pos_flat = positions.view(B, N, 2)
    out_orig_flat = out_orig.view(B, N, P, Q, F_dim)
    
    # Generate random permutation of the N cells
    perm = torch.randperm(N)
    inv_perm = torch.argsort(perm)
    
    X_perm = X_flat[:, perm].view(B, H, W, P, Q, F_dim)
    rf_perm = rf_flat[:, perm].view(B, H, W, R)
    pos_perm = pos_flat[:, perm].view(B, H, W, 2)
    
    out_perm = layer(X_perm, routing_features=rf_perm, positions=pos_perm)
    out_perm_flat = out_perm.view(B, N, P, Q, F_dim)
    
    # Unpermute
    out_reconstructed = out_perm_flat[:, inv_perm].view(B, H, W, P, Q, F_dim)
    
    diff = (out_orig - out_reconstructed).abs().max().item()
    assert diff < 1e-5, f"Permutation equivariance failure: max diff {diff} >= 1e-5"
