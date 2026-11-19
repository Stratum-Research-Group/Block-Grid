import math
import torch
import torch.nn as nn
from typing import Optional, Callable, Generator
from blockvga.tensor_cell_config import GeneralizedHyperTensorConfig
from blockvga.hyper_tensor_layer import GeneralizedHyperTensorLayer

class FlashHyperGridEngine(nn.Module):
    """
    FlashHyperGridEngine:
    Processes hypercube arrays beyond physical GPU VRAM by chunked streaming.
    Because landmark pooling is an associative accumulator:
        Z_k = sum_{chunk} sum_{n in chunk} w_{n, k} * V_n
    We stream chunks of N (e.g. 2^16 = 65,536 cells) sequentially into GPU VRAM,
    accumulating directly into slot latents Z without materializing the full 2^24+
    tensor in VRAM simultaneously.
    """
    def __init__(self, config: GeneralizedHyperTensorConfig, chunk_size: int = 65536, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.chunk_size = chunk_size # default 65,536 cells per chunk
        self.device = device
        self.core = GeneralizedHyperTensorLayer(config).to(device)

    @torch.inference_mode()
    def forward_streamed(self, chunk_generator: Callable[[], Generator[torch.Tensor, None, None]], total_N: int) -> torch.Tensor:
        """
        chunk_generator: callable yielding chunks of shape [B, chunk_size, *cell_tensor_shape]
        Returns:
            Z_trans: compact transformed latent slots [B, K, *cell_tensor_shape]
        """
        B = 1
        K = self.config.K
        cell_shape = self.config.cell_tensor_shape
        V_cell = self.config.cell_volume

        Z_accum_flat = torch.zeros((B, K, V_cell), device=self.device, dtype=torch.float32)

        for chunk_cpu in chunk_generator():
            # Transfer chunk to GPU
            chunk_gpu = chunk_cpu.to(self.device, non_blocking=True)
            cur_n = chunk_gpu.shape[1]
            flat_chunk = chunk_gpu.view(B, cur_n, V_cell)

            # Compute landmark weights for this chunk
            s_embed = self.core.summary_mlp(flat_chunk)
            w_write = torch.softmax((self.core.W_c(s_embed)) / 0.1, dim=-1) # [B, cur_n, K]

            # Accumulate into Z: einsum 'bnv,bnk->bkv'
            Z_accum_flat += torch.einsum('bnv,bnk->bkv', flat_chunk, w_write)
            del chunk_gpu, flat_chunk, s_embed, w_write

        # Slot mixing on compact latents Z (happens on K slots only, completely independent of N)
        Z_mix = torch.einsum('lk,bkv->blv', self.core.slotmix.get_mixing_matrix(), Z_accum_flat).view(B, K, *cell_shape)

        if self.core.cell_op is not None:
            Z_trans = self.core.cell_op(Z_mix)
        else:
            Z_trans = Z_mix

        return Z_trans
