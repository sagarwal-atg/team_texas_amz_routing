import torch
import math
from IPython import embed


class LinearModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.theta = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float32))

    def forward(self, x):
        return self.theta @ x


class ScalingLinearModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.theta = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float32))

    def forward(self, x):
        return torch.sigmoid(torch.mul(self.theta, x))


class DoubleLinearModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.theta_1 = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float32))
        self.theta_2 = torch.nn.Parameter(torch.randn((size, size), dtype=torch.float32))

    def forward(self, x):
        return self.theta_2 @ self.theta_1 @ x


class IRLLinearModel(torch.nn.Module):
    def __init__(self, max_route_len, link_features_size, route_features_size):
        super().__init__()
        self.link_features_size = link_features_size
        self.route_features_size = route_features_size
        self.theta = torch.nn.Parameter(torch.randn((max_route_len, max_route_len, link_features_size + route_features_size), dtype=torch.float32))

    def forward(self, input):
        output = torch.zeros((input.shape[0], input.shape[1], input.shape[2])).float()
        for i in range(input.shape[1]):
            for j in range(input.shape[2]):
                output[:, i, j] = input[:, i, j] @ self.theta[i, j]
        
        return output


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, max_route_len, link_features_size, route_features_size):
        super().__init__()
        self.link_features_size = link_features_size
        self.route_features_size = route_features_size
        self.linear = torch.nn.Linear(max_route_len * (link_features_size + route_features_size), max_route_len)

    def forward(self, input):
        """
        input : (batch_size, max_route_len, max_route_len, features)

        """
        output = torch.zeros((input.shape[0], input.shape[1], input.shape[2])).float()
        for stop_i in range(input.shape[1]):
            output[stop_i] 

        
        return output
