import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.optim = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        return self.mlp(x)

    def train_on_batch(self, inputs, labels):
        # zero the parameter gradients
        self.optim.zero_grad()

        # forward + backward + optimize
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optim.step()
        return loss

class ARC_Classifier(Classifier):
    """
    Amazon Routing Competition (ARC) Classifier
    """
    def __init__(self, max_route_len, num_features, hidden_sizes=[]):
        in_size = max_route_len * num_features
        super().__init__([in_size, *hidden_sizes, max_route_len])