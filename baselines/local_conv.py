import torch
import torch.nn as nn
from baselines.common import BaselineBase

class LocalConvBaseline(BaselineBase):
    """
    Local convolution stack with depth and receptive field sufficient to cover the full HxW grid.
    For H=W=16 (grid diameter 30), a stack of 15 layers of 3x3 conv covers the entire grid.
    """
    def __init__(self, H: int, W: int, P: int, Q: int, F_dim: int, R: int = 0, num_layers: int = 8):
        super().__init__(H, W, P, Q, F_dim, R)
        dim = self.payload_dim
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.GELU())
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor, routing_features=None, positions=None):
        B = X.shape[0]
        h = self.prepare_input(X, routing_features=routing_features, positions=positions) # [B, N, C]
        grid = h.view(B, self.H, self.W, self.payload_dim).permute(0, 3, 1, 2) # [B, C, H, W]
        
        out_conv = self.conv_stack(grid) # [B, C, H, W]
        out_h = out_conv.permute(0, 2, 3, 1).contiguous().view(B, self.H * self.W, self.payload_dim)
        return self.restore_output(out_h)
