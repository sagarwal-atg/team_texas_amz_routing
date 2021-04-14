import json
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset


LinkFeatures = namedtuple('LinkFeatures',
                          ['dist'])

RouteFeatures = namedtuple('RouteFeatures',
                          ['vehicle_cap',
                           'route_quality',
                           'num_packages'])

route_score_map = {'Low': 0, 'Medium': 1, 'High': 2}


class IRLDataset(Dataset):
    def __init__(self, config):
        with open(config.base_path + config.route_filepath) as f:
            route_data = json.load(f)

        with open(config.base_path + config.actual_filepath) as f:
            data_actual = json.load(f)

        with open(config.base_path + config.travel_times_filepath) as f:
            data_travel_time = json.load(f)
        
        with open(config.base_path + config.package_data_filepath) as f:
            package_data = json.load(f)
        
        route_ids = list(route_data.keys())
        datasize = config.datasize
        if config.datasize < 0:
            datasize = len(route_ids)

        self.link_features = LinkFeatures(dist=0)
        self.route_features = RouteFeatures(vehicle_cap=0, route_quality=1, num_packages=2)

        # features
        num_link_features = len(self.link_features)
        # package
        num_route_features = len(self.route_features)

        raw_link_data = np.zeros((datasize, config.max_route_len, config.max_route_len, num_link_features))
        raw_route_data = np.zeros((datasize, num_route_features))
        self.link_data = np.zeros((datasize, config.max_route_len, config.max_route_len, num_link_features))
        self.route_data = np.zeros((datasize, num_route_features))
        self.y = np.ones((datasize, config.max_route_len, config.max_route_len))

        self.stop_dict = dict()
        self.data_route_ids = []
        
        # Extract Features Vectors
        for route_number in range(datasize):
            stop_sequence = {k: v for k, v in data_actual[route_ids[route_number]]['actual'].items()}
                
            matrix_order = list(stop_sequence.keys())

            temp_dict = {}
            for i in range(len(matrix_order)):
                temp_dict[matrix_order[i]] = i

            distances = np.zeros((config.max_route_len, config.max_route_len))
            actual_utility = np.zeros((config.max_route_len, config.max_route_len))
            for i in matrix_order:
                for j in matrix_order:
                    distances[temp_dict[i]][temp_dict[j]] = data_travel_time[route_ids[route_number]][i][j]
                    if self.check_neighbor(stop_sequence[i], stop_sequence[j], len(stop_sequence)):
                        actual_utility[temp_dict[i]][temp_dict[j]] = 1
                
            self.y[route_number] = actual_utility

            raw_link_data[route_number, :, :, self.link_features.dist] = distances
            self.stop_dict[route_ids[route_number]] = stop_sequence 
            self.data_route_ids.append(route_ids[route_number])
        
        # Extract Package Data
        for route_number in range(datasize):
            vehicle_cap = route_data[route_ids[route_number]]['executor_capacity_cm3']
            route_score = route_data[route_ids[route_number]]['route_score']
            int_route_score = route_score_map[route_score]

            num_packages = 0
            package_data_for_route = package_data[route_ids[route_number]]
            for (stop, package_info) in package_data_for_route.items():
                num_packages += len(package_info)

            raw_route_data[route_number, self.route_features.vehicle_cap] = vehicle_cap
            raw_route_data[route_number, self.route_features.route_quality] = int_route_score
            raw_route_data[route_number, self.route_features.num_packages] = num_packages
        
        # Post Process Data
        scaler = MinMaxScaler()
        self.route_data = scaler.fit_transform(raw_route_data)

        norm = np.linalg.norm(raw_link_data[:, :, :, self.link_features.dist])
        self.link_data[:, :, :, self.link_features.dist] = raw_link_data[:, :, :, self.link_features.dist] / norm
        self.y = np.max(self.link_data[:, :, :, self.link_features.dist]) * \
            ( np.ones((datasize, config.max_route_len, config.max_route_len)) - self.y)

    def __len__(self):
        return self.link_data.shape[0]
    
    def check_neighbor(self, u, v, seq_len):
        is_neighbor = False
        if abs(u - v) == 1:
            is_neighbor = True
        elif abs(u - v) == seq_len:
            is_neighbor = True
        return is_neighbor

    def __getitem__(self, idx):
        x = np.zeros((self.link_data.shape[1], self.link_data.shape[2],
                      self.link_data.shape[-1] + self.route_data.shape[-1]))
        x[:, :, :self.link_data.shape[-1]] = self.link_data[idx]
        x[:, :, self.link_data.shape[-1]:] = self.route_data[idx]
        x = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(self.y[idx]).float()

        # orig_dist_matrix = self.orig_dist[idx]
        id = self.data_route_ids[idx]
        return (x, y_tensor, id)
