import torch
import torch.nn as nn
from typing import List
from blockvga.config import BlockGridConfig
from blockvga.layer import BlockGridLayer

class BlockGridModel(nn.Module):
    """
    Multi-layer BlockGrid network with optional classification head or projection.
    """
    def __init__(self, config: BlockGridConfig, num_layers: int = 2, num_classes: int = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([BlockGridLayer(config) for _ in range(num_layers)])
        
        if num_classes is not None:
            # Classification head: mean-pool over (H, W, P, Q) -> Linear(F, num_classes)
            self.head = nn.Linear(config.F, num_classes)
        else:
            self.head = None

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        out = X
        for layer in self.layers:
            out = layer(out, routing_features=routing_features, positions=positions)
            
        if self.head is not None:
            # Global mean pool across spatial & block dimensions
            pooled = out.mean(dim=(1, 2, 3, 4))  # [B, F]
            return self.head(pooled)
        return out
