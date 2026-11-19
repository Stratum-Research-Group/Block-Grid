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

    Note:
        In streaming mode without pre-computed or streamed coordinate buffers,
        the engine uses content-based landmark routing with learned/calibrated
        temperature tau. If positions are not yielded by the generator, routing
        operates on cell content embeddings.
    """
    def __init__(self, config: GeneralizedHyperTensorConfig, chunk_size: int = 65536, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.chunk_size = chunk_size # default 65,536 cells per chunk
        self.device = device
        self.core = GeneralizedHyperTensorLayer(config).to(device)

    @torch.inference_mode()
    def forward_streamed(
        self,
        chunk_generator: Callable[[], Generator[torch.Tensor, None, None]],
        total_N: Optional[int] = None,
        positions_generator: Optional[Callable[[], Generator[torch.Tensor, None, None]]] = None,
    ) -> torch.Tensor:
        """
        chunk_generator: callable yielding chunks of shape [B, chunk_size, *cell_tensor_shape] (or [chunk_size, *cell_tensor_shape])
        total_N: total number of cells across all chunks (optional, unused for streaming accumulation)
        positions_generator: optional callable yielding matching coordinate chunks [B, chunk_size, D]
        Returns:
            Z_trans: compact transformed latent slots [B, K, *cell_tensor_shape]
        """
        K = self.config.K
        cell_shape = self.config.cell_tensor_shape
        V_cell = self.config.cell_volume

        Z_accum_flat = None
        B = None

        pos_iter = iter(positions_generator()) if positions_generator is not None else None

        for chunk_cpu in chunk_generator():
            # Standardize batch dimension
            if chunk_cpu.dim() == len(cell_shape) + 1:
                chunk_cpu = chunk_cpu.unsqueeze(0)
            cur_b, cur_n = chunk_cpu.shape[0], chunk_cpu.shape[1]
            if B is None:
                B = cur_b
                Z_accum_flat = torch.zeros((B, K, V_cell), device=self.device, dtype=torch.float32)

            # Transfer chunk to GPU
            chunk_gpu = chunk_cpu.to(self.device, non_blocking=True)
            flat_chunk = chunk_gpu.view(B, cur_n, V_cell)

            # Compute landmark weights for this chunk
            s_embed = self.core.summary_mlp(flat_chunk)
            logits = self.core.W_c(s_embed)

            if pos_iter is not None:
                pos_cpu = next(pos_iter)
                if pos_cpu.dim() == 2:
                    pos_cpu = pos_cpu.unsqueeze(0)
                pos_gpu = pos_cpu.to(self.device, non_blocking=True)
                p_embed = 0.0
                for i, embed_layer in enumerate(self.core.axis_embeds):
                    p_embed = p_embed + embed_layer(pos_gpu[..., i])
                logits = logits + self.core.W_p(p_embed)
                del pos_gpu, p_embed

            w_write = self.core._compute_simplex(logits, flat_chunk.dtype)

            # Accumulate into Z: einsum 'bnv,bnk->bkv'
            Z_accum_flat += torch.einsum('bnv,bnk->bkv', flat_chunk, w_write)
            del chunk_gpu, flat_chunk, s_embed, logits, w_write

        if Z_accum_flat is None:
            B = 1
            Z_accum_flat = torch.zeros((B, K, V_cell), device=self.device, dtype=torch.float32)

        # Slot mixing on compact latents Z (happens on K slots only, completely independent of N)
        Z_mix = torch.einsum('lk,bkv->blv', self.core.slotmix.get_mixing_matrix(), Z_accum_flat).view(B, K, *cell_shape)

        if self.core.cell_op is not None:
            Z_trans = self.core.cell_op(Z_mix)
        else:
            Z_trans = Z_mix

        return Z_trans
