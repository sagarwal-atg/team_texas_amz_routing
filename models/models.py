from os import link
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class LinearModel(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.theta = torch.randn((size), dtype=torch.float32)

    def forward(self, link_features, route_features):
        link_features = torch.Tensor(link_features)
        travel_time_matrix = torch.zeros((link_features.shape[0], link_features.shape[1]))

        # route_constant = self.theta[-1] * torch.Tensor(route_features)
        for i in range(link_features.shape[0]):
            for j in range(link_features.shape[1]):
                travel_time_matrix[i][j] = self.theta.T @ link_features[i][j]

        return travel_time_matrix


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Classifier(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.mlp = mlp(sizes)

        # This loss fn combines LogSoftmax and NLLLoss in one single class
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.mlp(x)

    def get_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)


class ARC_Classifier(Classifier):
    """
    Amazon Routing Competition (ARC) Classifier
    """
    def __init__(self, max_route_len, num_features, hidden_sizes=[]):
        in_size = max_route_len * num_features
        sizes=[in_size, *hidden_sizes, max_route_len]
        super().__init__(sizes)