from os import link

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


class IRLModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.fc3 = nn.Linear(num_features, 1)

        param = torch.FloatTensor((0.24744)).type(torch.FloatTensor)
        self.lamb = torch.nn.Parameter(param, requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_hat = self.fc3(x)
        return y_hat
