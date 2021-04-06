import json
import numpy as np

import torch
from torch.utils.data import Dataset


class RouteDataset(Dataset):
    def __init__(self, route_filepath, actual_filepath, travel_times_filepath, MAX_ROUTE_LEN, datasize=-1, MAX_COST=5000.0):
        with open(route_filepath) as f:
            route_data = json.load(f)

        with open(actual_filepath) as f:
            data_actual = json.load(f)

        with open(travel_times_filepath) as f:
            data_travel_time = json.load(f)
        
        route_ids = list(route_data.keys())

        if datasize < 0:
            datasize = len(route_ids)

        self.x = np.zeros((datasize, MAX_ROUTE_LEN, MAX_ROUTE_LEN))
        ## @TODO fix max cost
        self.y = float(MAX_COST) * np.ones((datasize, MAX_ROUTE_LEN, MAX_ROUTE_LEN))

        self.stop_dict = dict()
        self.data_route_ids = []
        
        for route_number in range(datasize):
            stop_sequence = {k: v for k, v in data_actual[route_ids[route_number]]['actual'].items()}
                
            matrix_order = list(stop_sequence.keys())

            temp_dict = {}
            for i in range(len(matrix_order)):
                temp_dict[matrix_order[i]] = i

            distances = np.zeros((MAX_ROUTE_LEN, MAX_ROUTE_LEN))
            actual_utility = np.zeros((MAX_ROUTE_LEN, MAX_ROUTE_LEN))
            for i in matrix_order:
                for j in matrix_order:
                    distances[temp_dict[i]][temp_dict[j]] = data_travel_time[route_ids[route_number]][i][j]
                    if self.check_neighbor(stop_sequence[i], stop_sequence[j], len(stop_sequence)):
                        actual_utility[temp_dict[i]][temp_dict[j]] = 1
                
            self.y[route_number] = self.y[route_number] - float(MAX_COST) * actual_utility

            self.x[route_number] = distances
            self.stop_dict[route_ids[route_number]] = stop_sequence 
            self.data_route_ids.append(route_ids[route_number])
    
    def check_neighbor(self, u, v, seq_len):
        is_neighbor = False
        if abs(u - v) == 1:
            is_neighbor = True
        elif abs(u - v) == seq_len:
            is_neighbor = True
        return is_neighbor

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_tensor = torch.from_numpy(self.x[idx])
        y_tensor = torch.from_numpy(self.y[idx])
        id = self.data_route_ids[idx]
        return (x_tensor, y_tensor, id)
