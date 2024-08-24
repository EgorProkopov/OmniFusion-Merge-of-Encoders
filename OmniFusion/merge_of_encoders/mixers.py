import torch
import torch.nn as nn

class WeightedSumMixer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.weight_1 = nn.Parameter(torch.randn(1,))
        self.weight_2 = nn.Parameter(torch.randn(1,))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z = self.weight_1 * x1 + self.weight_2 + x2
        return z