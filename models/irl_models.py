from os import link

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


class IRLModel(nn.Module):
    def __init__(self, num_features, out_features=1):
        super().__init__()

        init_mean = 0.0
        init_std = 0.5

        self.fc1 = nn.Linear(num_features, num_features)
        nn.init.normal_(self.fc1.weight, mean=init_mean, std=init_std)

        self.fc2 = nn.Linear(num_features, int(num_features / 2))
        nn.init.normal_(self.fc2.weight, mean=init_mean, std=init_std)

        self.fc3 = nn.Linear(int(num_features / 2), out_features)
        nn.init.normal_(self.fc3.weight, mean=init_mean, std=init_std)

        param = torch.FloatTensor([1.24744]).type(torch.FloatTensor)
        self.lamb = torch.nn.Parameter(param, requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = torch.sigmoid(x) + 0.5
        return x

    def get_lambda(self):
        return torch.clamp(self.lamb, min=1.0)


class IRL_Neighbor_Model(nn.Module):
    def __init__(self, num_features, num_base_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_features - (1 * num_base_features))
        nn.init.normal_(self.fc1.weight, mean=1, std=0.1)

        self.fc2 = nn.Linear(
            num_features - (1 * num_base_features),
            num_features - (2 * num_base_features),
        )
        nn.init.normal_(self.fc2.weight, mean=1, std=0.1)

        self.fc3 = nn.Linear(num_features - (2 * num_base_features), 1)
        nn.init.normal_(self.fc3.weight, mean=1, std=0.1)

        param = torch.FloatTensor([1.24744]).type(torch.FloatTensor)
        self.lamb = torch.nn.Parameter(param, requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def get_lambda(self):
        return torch.clamp(self.lamb, min=0.2)


class TC_Model(nn.Module):
    def __init__(self, num_features, out_features=1):
        super().__init__()

        init_mean = 1.0
        init_std = 0.1

        self.fc1 = nn.Linear(num_features, num_features)
        nn.init.normal_(self.fc1.weight, mean=init_mean, std=init_std)

        self.fc2 = nn.Linear(num_features, int(num_features / 2))
        nn.init.normal_(self.fc2.weight, mean=init_mean, std=init_std)

        self.fc3 = nn.Linear(int(num_features / 2), out_features)
        nn.init.normal_(self.fc3.weight, mean=init_mean, std=init_std)

        param = torch.FloatTensor([1.24744]).type(torch.FloatTensor)
        self.lamb = torch.nn.Parameter(param, requires_grad=True)

    def forward(self, x, curr_len):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x[:, 0] = torch.sigmoid(x[:, 0]) * 8
        x[:, 1] = torch.sigmoid(x[:, 1]) * curr_len
        return x

    def get_lambda(self):
        return torch.clamp(self.lamb, min=1.0)
