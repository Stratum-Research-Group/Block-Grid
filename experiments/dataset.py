import math
import random
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticBlockTransportDataset(Dataset):
    """
    Stage 1 - Synthetic block transport task:
    H = W = 16 grid (N = 256 cells), P = Q = 8, F = 4.
    Per episode:
      - Payload init with background noise N(0, 0.1)
      - One random source cell receives random target pattern N(0, 1)
      - One random destination cell (distinct from source)
      - Markers in routing_features [B, H, W, 2]: ch 0 = 1 at src, ch 1 = 1 at dst
      - Task 1a: target is src pattern at dst cell.
      - Task 1b: target is A* V_src B*^T for fixed A*, B*.
    """
    def __init__(self, num_episodes: int, task: str = "1a", A_star=None, B_star=None,
                 min_dist: int = 1, max_dist: int = 30, seed: int = 42,
                 H: int = 16, W: int = 16, P: int = 8, Q: int = 8, F_dim: int = 4):
        self.num_episodes = num_episodes
        self.task = task
        self.A_star = A_star
        self.B_star = B_star
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.seed = seed
        self.H = H
        self.W = W
        self.P = P
        self.Q = Q
        self.F_dim = F_dim
        
        # Pre-generate episodes or use fixed seed RNG
        self.rng = random.Random(seed)
        self.torch_rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        # Background noise N(0, 0.1)
        X = torch.randn(self.H, self.W, self.P, self.Q, self.F_dim, generator=self.torch_rng) * 0.1
        
        # Select source and destination satisfying distance constraint
        while True:
            si, sj = self.rng.randint(0, self.H - 1), self.rng.randint(0, self.W - 1)
            di, dj = self.rng.randint(0, self.H - 1), self.rng.randint(0, self.W - 1)
            dist = abs(si - di) + abs(sj - dj)
            if (si != di or sj != dj) and (self.min_dist <= dist <= self.max_dist):
                break
                
        # Target pattern N(0, 1)
        src_pattern = torch.randn(self.P, self.Q, self.F_dim, generator=self.torch_rng)
        X[si, sj] = src_pattern
        
        # Routing features: [H, W, 2]
        rf = torch.zeros(self.H, self.W, 2)
        rf[si, sj, 0] = 1.0
        rf[di, dj, 1] = 1.0
        
        # Compute ground truth target at destination
        if self.task == "1a":
            target_dst = src_pattern
        elif self.task == "1b":
            # A* V_src B*^T
            # 'ip, jq, pqf -> ijf'
            target_dst = torch.einsum('ip,jq,pqf->ijf', self.A_star, self.B_star, src_pattern)
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        return X, rf, (di, dj), target_dst, dist
