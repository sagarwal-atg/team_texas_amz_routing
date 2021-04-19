import json
from typing import Sequence
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
import enum

import torch
from torch.utils.data import Dataset

class LinkFeatures(enum.Enum):
    Distance = 0
    # add additional features here along with their index

class RouteFeatures(enum.Enum):
    VehicleCapacity = 0
    RouteQuality = 1
    NumPackages = 2
    # add additional features here along with their index


route_score_map = {'Low': 0, 'Medium': 1, 'High': 2}


def check_neighbor(u: int, v: int, seq_len: int):
    # return true if we go from u to v
    return v - u == 1 or v - u == seq_len


class IRLDataset(Dataset):
    def __init__(self, route_path, label_path, travel_times_path, packages_path, max_route_len=None):
        with open(route_path) as f:
            route_data = json.load(f)

        with open(label_path) as f:
            data_actual = json.load(f)

        with open(travel_times_path) as f:
            data_travel_time = json.load(f)

        with open(packages_path) as f:
            package_data = json.load(f)

        route_ids = list(route_data.keys())
        n_routes = len(route_ids)
        route_lengths = [len(data_actual[route_id]['actual']) for route_id in route_ids]

        if max_route_len is None:
            max_route_len = max(route_lengths)
        self.max_route_len = max_route_len

        self.num_link_features = len(LinkFeatures)
        self.num_route_features = len(RouteFeatures)
        self.num_features = self.num_link_features + self.num_route_features

        raw_link_data = []
        route_taken_matrices = []
        raw_route_data = np.zeros((n_routes, self.num_route_features))

        # Extract Features Vectors
        for route_number, (route_id, route_len) in enumerate(zip(route_ids, route_lengths)):
            stop_sequence = data_actual[route_id]['actual']

            stop_ids = list(stop_sequence.keys())

            stop_idx = {k: i for i, k in enumerate(stop_ids)}

            distances = np.zeros((route_len, max_route_len))
            route_matrix = np.zeros((route_len, max_route_len))
            for i in stop_ids:
                for j in stop_ids:
                    distances[stop_idx[i]][stop_idx[j]] = data_travel_time[route_id][i][j]
                    if check_neighbor(stop_sequence[i], stop_sequence[j], len(stop_sequence)):
                        route_matrix[stop_idx[i]][stop_idx[j]] = 1

            route_taken_matrices.append(route_matrix)

            link_features = np.dstack([distances])
            raw_link_data.append(link_features)

        # Extract route data
        for route_number in range(n_routes):
            vehicle_cap = route_data[route_id]['executor_capacity_cm3']
            route_score = route_data[route_id]['route_score']
            int_route_score = route_score_map[route_score]

            num_packages = 0
            package_data_for_route = package_data[route_id]
            for (_, package_info) in package_data_for_route.items():
                num_packages += len(package_info)

            raw_route_data[route_number, RouteFeatures.VehicleCapacity.value] = vehicle_cap
            raw_route_data[route_number, RouteFeatures.RouteQuality.value] = int_route_score
            raw_route_data[route_number, RouteFeatures.NumPackages.value] = num_packages

        # shape(route_date)=[n_routes, num_route_features]
        route_data = np.repeat(raw_route_data, route_lengths, 0)
        # shape(route_date)=[sum(route_lengths), num_route_features]
        route_data = np.tile(route_data[:,np.newaxis,:], [1,max_route_len,1])
        # shape(route_date)=[sum(route_lengths), max_route_len, num_route_features]
        link_data = np.concatenate(raw_link_data)
        x = np.dstack([route_data, link_data])
        # shape(x)=[sum(route_lengths), max_route_len, num_features]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        # shape(x)=[sum(route_lengths), max_route_len * num_features]
        y = np.concatenate(route_taken_matrices)

        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(np.argmax(y, axis=1)) # convert one-hot-encoding to list of indexes

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
