import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blockgrid.config import BlockGridConfig

class LandmarkAssignment(nn.Module):
    """
    Step 1 & 2: Cell summary projection & landmark assignment simplex.
    """
    def __init__(self, config: BlockGridConfig):
        super().__init__()
        self.config = config
        H, W, P, Q, F_dim, d, K, R = (
            config.H, config.W, config.P, config.Q, config.F, config.d, config.K, config.R
        )
        
        # Positional embeddings
        self.row_embed = nn.Embedding(H, d)
        self.col_embed = nn.Embedding(W, d)
        
        # Summary MLP
        if config.summary == "mean":
            self.summary_mlp = nn.Sequential(
                nn.Linear(F_dim, d),
                nn.GELU(),
                nn.Linear(d, d)
            )
        elif config.summary == "structured":
            self.summary_mlp = nn.Sequential(
                nn.Linear(P * Q * F_dim, d),
                nn.GELU(),
                nn.Linear(d, d)
            )
        else:
            raise ValueError(f"Unknown summary mode: {config.summary}")
            
        # Optional routing features projection
        if R > 0:
            self.linear_r = nn.Linear(R, d)
        else:
            self.linear_r = None
            
        # Routing projections to K slots (or num_heads * slots_per_head)
        self.num_heads = config.num_heads
        if config.K % config.num_heads != 0:
            raise ValueError(f"K ({config.K}) must be divisible by num_heads ({config.num_heads})")
        self.slots_per_head = config.K // config.num_heads

        if "pos" not in config.assignment and "content" not in config.assignment:
            raise ValueError(
                f"Invalid assignment mode '{config.assignment}'. Must contain 'pos' or 'content' (e.g. 'pos+content', 'pos', 'content')"
            )

        self.W_p = nn.Linear(d, K, bias=False)
        self.W_c = nn.Linear(d, K, bias=False)
        
        # Learnable temperature
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(config.tau_init, 0.05))))
        
        # Untied readback projection if enabled
        if not config.tied_readback:
            self.W_p_read = nn.Linear(d, K, bias=False)
            self.W_c_read = nn.Linear(d, K, bias=False)
        else:
            self.W_p_read = None
            self.W_c_read = None

    def get_temperature(self) -> torch.Tensor:
        tau = torch.exp(self.log_tau)
        return torch.clamp(tau, min=0.05)

    def _compute_simplex(self, logits: torch.Tensor, V_dtype: torch.dtype) -> torch.Tensor:
        # logits: [B, H, W, K]
        # Reshape to [B, H, W, num_heads, slots_per_head]
        B, H, W, K = logits.shape
        tau = self.get_temperature()
        logits_reshaped = logits.view(B, H, W, self.num_heads, self.slots_per_head)
        
        if V_dtype in (torch.float16, torch.bfloat16):
            w = F.softmax((logits_reshaped / tau).float(), dim=-1).to(V_dtype)
        else:
            w = F.softmax(logits_reshaped / tau, dim=-1)
            
        return w.view(B, H, W, K)

    def forward(self, V, routing_features=None, positions=None):
        """
        V: [B, H, W, P, Q, F]
        routing_features: [B, H, W, R] or None
        positions: [B, H, W, 2] integer indices or None
        Returns:
            w_write: [B, H, W, K]
            w_read:  [B, H, W, K]
        """
        B, H, W, P, Q, F_dim = V.shape
        d = self.config.d
        device = V.device
        
        # Positions
        if positions is None:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            row_idx = grid_y.unsqueeze(0).expand(B, -1, -1)
            col_idx = grid_x.unsqueeze(0).expand(B, -1, -1)
        else:
            if not torch.is_floating_point(positions):
                pass
            else:
                raise ValueError("Positions must be integer coordinate indices.")
            if (positions[..., 0] < 0).any() or (positions[..., 0] >= self.config.H).any():
                raise ValueError(f"Row position out of bounds [0, {self.config.H})")
            if (positions[..., 1] < 0).any() or (positions[..., 1] >= self.config.W).any():
                raise ValueError(f"Col position out of bounds [0, {self.config.W})")
            row_idx = positions[..., 0]
            col_idx = positions[..., 1]
            
        p_a = self.row_embed(row_idx) + self.col_embed(col_idx)  # [B, H, W, d]
        
        # Cell summary
        if self.config.summary == "mean":
            mean_v = V.mean(dim=(-3, -2))  # [B, H, W, F]
            s_a = self.summary_mlp(mean_v)  # [B, H, W, d]
        else:
            flat_v = V.flatten(start_dim=-3)  # [B, H, W, P*Q*F]
            s_a = self.summary_mlp(flat_v)  # [B, H, W, d]
            
        if self.linear_r is not None:
            if routing_features is None:
                raise ValueError(f"routing_features must be provided when R={self.config.R} > 0")
            s_a = s_a + self.linear_r(routing_features)
            
        # Compute logits for write
        logits_write = 0.0
        if "pos" in self.config.assignment:
            logits_write = logits_write + self.W_p(p_a)
        if "content" in self.config.assignment:
            logits_write = logits_write + self.W_c(s_a)
            
        w_write = self._compute_simplex(logits_write, V.dtype)
        
        if self.config.tied_readback:
            w_read = w_write
        else:
            logits_read = 0.0
            if "pos" in self.config.assignment:
                logits_read = logits_read + self.W_p_read(p_a)
            if "content" in self.config.assignment:
                logits_read = logits_read + self.W_c_read(s_a)
            w_read = self._compute_simplex(logits_read, V.dtype)
            
        return w_write, w_read
