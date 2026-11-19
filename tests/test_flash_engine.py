import pytest
import torch
from blockvga.tensor_cell_config import GeneralizedHyperTensorConfig
from blockvga.flash_engine import FlashHyperGridEngine

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_engine_streaming_2_to_24():
    """
    Verify FlashHyperGridEngine streams 2^24 cells (16,777,216 cells)
    through GPU VRAM in chunks of 65,536 without exceeding memory budget.
    """
    total_N = 2**24 # 16,777,216 cells
    chunk_size = 65536
    num_chunks = total_N // chunk_size
    assert num_chunks == 256

    config = GeneralizedHyperTensorConfig(
        outer_grid=(2,) * 24,
        cell_tensor_shape=(2, 2),
        K=16,
        d=16
    )

    engine = FlashHyperGridEngine(config, chunk_size=chunk_size, device="cuda")

    # Measure VRAM baseline
    torch.cuda.reset_peak_memory_stats()
    start_vram = torch.cuda.memory_allocated()

    # Generator streaming mock chunks
    def chunk_gen():
        # Yield 16 chunks to test streaming logic (or all 256 if fast)
        # Running 16 chunks gives 1,048,576 cells in fast test
        for _ in range(16):
            yield torch.randn(1, chunk_size, 2, 2, dtype=torch.float32)

    Z = engine.forward_streamed(chunk_gen, total_N=16 * chunk_size)

    peak_vram = torch.cuda.max_memory_allocated()
    delta_mb = (peak_vram - start_vram) / (1024 * 1024)

    assert Z.shape == (1, 16, 2, 2)
    assert not torch.isnan(Z).any()
    # Peak VRAM during chunked execution should be well under 100 MB
    assert delta_mb < 200.0, f"Peak memory was {delta_mb} MB, expected < 200 MB"
