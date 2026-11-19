import torch
import torch.nn as nn
from blockgrid.config import BlockGridConfig
from blockgrid.slotmix import SlotMixing

def test_planar_uncrossed_mixing():
    """
    Verify planar uncrossed identity parameterization G = I - U V^T
    """
    K = 16
    cfg = BlockGridConfig(H=4, W=4, P=2, Q=2, F=4, K=K, slot_mix="planar_uncrossed")
    slot = SlotMixing(cfg)
    G = slot.get_mixing_matrix()
    
    assert G.shape == (K, K)
    # Check that it's close to identity initially (init weights ~ 0.02)
    diag = torch.diag(G)
    assert (diag - 1.0).abs().mean() < 0.1
    # Check that gradient flows to U and V
    loss = G.sum()
    loss.backward()
    assert slot.U.grad is not None and slot.U.grad.norm().item() > 0
    assert slot.V.grad is not None and slot.V.grad.norm().item() > 0
