import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blockgrid.tensor_cell_config import GeneralizedHyperTensorConfig
from blockgrid.slotmix import SlotMixing
from blockgrid.config import BlockGridConfig

class CellTensorOperator(nn.Module):
    """
    Applies multi-linear multilinear contractions on an arbitrary inner tensor (T_1, T_2, ..., T_M).
    For each mode m in 1..M, applies mode-m matrix product:
        Z' = Z x_1 A^(1) x_2 A^(2) ... x_M A^(M)
    where A^(m) in R^{K x T_m x T_m}.
    """
    def __init__(self, cell_tensor_shape, K: int):
        super().__init__()
        self.cell_tensor_shape = cell_tensor_shape
        self.K = K
        self.mode_transforms = nn.ParameterList()
        
        for dim_size in cell_tensor_shape:
            # Init mode transform A_k: [K, dim_size, dim_size] as Identity + N(0, 0.01)
            init_A = torch.eye(dim_size).unsqueeze(0).repeat(K, 1, 1) + torch.randn(K, dim_size, dim_size) * 0.01
            self.mode_transforms.append(nn.Parameter(init_A))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [B, K, T_1, T_2, ..., T_M]
        Returns: [B, K, T_1, T_2, ..., T_M]
        """
        out = Z
        M = len(self.cell_tensor_shape)
        # Contract along each mode m
        for m, A in enumerate(self.mode_transforms):
            # A has shape [K, T_m, T_m]
            # out has shape [B, K, T_1, ..., T_m, ..., T_M]
            # We want: out[b, k, ..., i, ...] = sum_{j} A[k, i, j] * out[b, k, ..., j, ...]
            # Move mode m to the end:
            # dim 0: B, dim 1: K, then remaining M dimensions
            perm = list(range(out.dim()))
            target_axis = 2 + m
            perm.append(perm.pop(target_axis)) # move mode m to end
            out_perm = out.permute(*perm) # [B, K, remaining..., T_m]
            
            # Contraction: '...j, kij -> ...i'
            # Using matmul: out_perm is [..., T_m], A is [K, T_m, T_m]
            # Since A is [K, T_m, T_m], A.transpose(-1, -2) is [K, T_m, T_m]
            # out_new = torch.matmul(out_perm, A.transpose(-1, -2))
            # Reshape A to broadcast over leading batch dims if needed
            # More direct via einsum:
            # We can use einsum with dynamic letters:
            letters = "cdefghijklmnopqrstuvwxyz"[:M]
            orig_letters = letters
            new_letter = "a"
            trans_letter = letters[m]
            res_letters = letters[:m] + new_letter + letters[m+1:]
            
            # eqn: 'bk' + orig_letters + ', k' + new_letter + trans_letter + ' -> bk' + res_letters
            eqn = f"bk{orig_letters},k{new_letter}{trans_letter}->bk{res_letters}"
            out = torch.einsum(eqn, out, A)
            
        return out

class GeneralizedHyperTensorLayer(nn.Module):
    """
    Operates on a hyperdimensional grid where EVERY cell holds an arbitrary tensor:
    X: [B, S_1, ..., S_D, T_1, ..., T_M]
    """
    def __init__(self, config: GeneralizedHyperTensorConfig):
        super().__init__()
        self.config = config
        d = config.d
        K = config.K

        self.num_heads = config.num_heads
        if K % self.num_heads != 0:
            raise ValueError(f"K ({K}) must be divisible by num_heads ({self.num_heads})")
        self.slots_per_head = K // self.num_heads

        # Axis embedding tables for outer hyper-coordinates
        self.axis_embeds = nn.ModuleList([
            nn.Embedding(s, d) for s in config.outer_grid
        ])

        # Cell summary MLP over the entire inner tensor volume
        self.summary_mlp = nn.Sequential(
            nn.Linear(config.cell_volume, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

        self.W_p = nn.Linear(d, K, bias=False)
        self.W_c = nn.Linear(d, K, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(config.tau_init, 0.05))))

        if not config.tied_readback:
            self.W_p_read = nn.Linear(d, K, bias=False)
            self.W_c_read = nn.Linear(d, K, bias=False)
        else:
            self.W_p_read = None
            self.W_c_read = None

        # Slot mixing across K slots
        dummy_bg = BlockGridConfig(H=1, W=config.N, P=1, Q=1, F=1, K=K, slot_mix=config.slot_mix)
        self.slotmix = SlotMixing(dummy_bg)

        # Multi-mode tensor operator for inner cell tensors
        if config.inner_transform:
            self.cell_op = CellTensorOperator(config.cell_tensor_shape, K)
        else:
            self.cell_op = None

        # Residual projection over cell volume
        self.cell_proj = nn.Linear(config.cell_volume, config.cell_volume)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def _compute_simplex(self, logits: torch.Tensor, V_dtype: torch.dtype) -> torch.Tensor:
        B, N, K = logits.shape
        tau = torch.clamp(torch.exp(self.log_tau), min=0.05)
        logits_reshaped = logits.view(B, N, self.num_heads, self.slots_per_head)
        if V_dtype in (torch.float16, torch.bfloat16):
            w = F.softmax((logits_reshaped / tau).float(), dim=-1).to(V_dtype)
        else:
            w = F.softmax(logits_reshaped / tau, dim=-1)
        return w.view(B, N, K)

    def forward(self, X: torch.Tensor, positions=None):
        """
        X: [B, S_1, ..., S_D, T_1, ..., T_M] or [B, N, T_1, ..., T_M]
        """
        B = X.shape[0]
        N = self.config.N
        V_cell = self.config.cell_volume
        cell_shape = self.config.cell_tensor_shape
        orig_shape = X.shape

        # Flatten outer hypercube axes into N, keeping inner tensor intact
        # [B, N, T_1, ..., T_M]
        X_reshaped = X.view(B, N, *cell_shape)
        device = X.device

        # 1. Coordinate embeddings
        if positions is None:
            grids = torch.meshgrid(
                *[torch.arange(s, device=device) for s in self.config.outer_grid],
                indexing='ij'
            )
            pos = torch.stack([g.flatten() for g in grids], dim=-1).unsqueeze(0).expand(B, -1, -1)
        else:
            pos = positions

        p_embed = 0.0
        for i, embed_layer in enumerate(self.axis_embeds):
            p_embed = p_embed + embed_layer(pos[..., i])

        # Cell summary over the inner tensor
        flat_cells = X_reshaped.view(B, N, V_cell)
        s_embed = self.summary_mlp(flat_cells)

        logits_write = self.W_p(p_embed) + self.W_c(s_embed)
        w_write = self._compute_simplex(logits_write, X.dtype)

        if self.config.tied_readback:
            w_read = w_write
        else:
            logits_read = self.W_p_read(p_embed) + self.W_c_read(s_embed)
            w_read = self._compute_simplex(logits_read, X.dtype)

        # 2. Pool whole cell tensors into K slots:
        # Einsum over N: X_flat [B, N, V_cell], w_write [B, N, K] -> Z_flat [B, K, V_cell]
        Z_flat = torch.einsum('bnv,bnk->bkv', flat_cells, w_write)
        Z = Z_flat.view(B, self.config.K, *cell_shape)

        # 3. Slot mixing
        # Reshape Z for slot mixing: [B, K, V_cell]
        Z_mix = torch.einsum('lk,bkv->blv', self.slotmix.get_mixing_matrix(), Z_flat).view(B, self.config.K, *cell_shape)

        # 4. Multilinear Mode Contraction on each inner cell tensor
        if self.cell_op is not None:
            Z_transformed = self.cell_op(Z_mix)
        else:
            Z_transformed = Z_mix

        # 5. Readback to all hyper-coordinates
        Z_tf_flat = Z_transformed.view(B, self.config.K, V_cell)
        Y_flat = torch.einsum('bkv,bnk->bnv', Z_tf_flat, w_read)

        # 6. Residual combine
        out_flat = flat_cells + self.gamma * self.cell_proj(Y_flat)
        return out_flat.view(orig_shape)
