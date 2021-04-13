import torch
import math
from IPython import embed


class LinearModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.theta = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float64))

    def forward(self, x):
        return self.theta @ x


class ScalingLinearModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.theta = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float64))

    def forward(self, x):
        return torch.sigmoid(torch.mul(self.theta, x))


class DoubleLinearModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.theta_1 = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float64))
        self.theta_2 = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float64))

    def forward(self, x):
        return self.theta_2 @ self.theta_1 @ x