import torch
import math
from IPython import embed


class LinearModel(torch.nn.Module):
    def __init__(self, size):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.theta = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float64))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.theta @ x