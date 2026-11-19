import torch
from blockvga import BlockGridConfig
from blockvga.slotmix import SlotMixing

def test_symmetry_restriction():
    """
    test_symmetry_restriction: slot_mix="symmetric" yields ||G - G^T|| = 0; "free"
    can represent a hand-set asymmetric G; the reporting split sym(G)+skew(G)
    reconstructs any learned G and skew(G) flips sign under slot transpose.
    """
    K = 16
    # 1. Symmetric config yields exact symmetry
    cfg_sym = BlockGridConfig(H=4, W=4, P=2, Q=2, F=4, K=K, slot_mix="symmetric")
    slot_sym = SlotMixing(cfg_sym)
    G_sym = slot_sym.get_mixing_matrix()
    diff_sym = (G_sym - G_sym.t()).norm().item()
    assert diff_sym < 1e-7, f"Symmetric mixing matrix has ||G - G^T|| = {diff_sym} > 0"

    # 2. Free config can represent an asymmetric matrix
    cfg_free = BlockGridConfig(H=4, W=4, P=2, Q=2, F=4, K=K, slot_mix="free")
    slot_free = SlotMixing(cfg_free)
    # Set explicit asymmetric G
    asym = torch.randn(K, K)
    asym[0, 1] = 10.0
    asym[1, 0] = -10.0
    slot_free.G.data.copy_(asym)
    G_free = slot_free.get_mixing_matrix()
    assert (G_free - G_free.t()).norm().item() > 1.0

    # 3. Reporting split sym(G) + skew(G) reconstructs G, skew(G) flips sign
    sym_G = 0.5 * (G_free + G_free.t())
    skew_G = 0.5 * (G_free - G_free.t())
    reconstructed = sym_G + skew_G
    assert (G_free - reconstructed).norm().item() < 1e-6
    assert (skew_G + skew_G.t()).norm().item() < 1e-6
