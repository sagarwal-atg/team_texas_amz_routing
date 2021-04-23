import json
import numpy as np

import torch
from torch.utils.data import Dataset


class DistMatrixDataset():
    def __init__(self, data_config, route_ids):
        with open(data_config.route_path) as f:
            route_data = json.load(f)

        with open(data_config.sequence_path) as f:
            data_actual = json.load(f)

        with open(data_config.travel_time_path) as f:
            data_travel_time = json.load(f)
        
        self.tt_matrices = []
        self.stops = []
        self.travel_times_dict = []

        self.stop_dict = dict()
        self.data_route_ids = []
        
        for route_number in range(len(route_ids)):
            stop_sequence = {k: v for k, v in data_actual[route_ids[route_number]]['actual'].items()}
                
            matrix_order = list(stop_sequence.keys())
            self.stops.append(stop_sequence)

            temp_dict = {}
            for i in range(len(matrix_order)):
                temp_dict[matrix_order[i]] = i

            distances = np.zeros((len(matrix_order), len(matrix_order)))
            self.travel_times_dict.append(data_travel_time[route_ids[route_number]])
            for i in matrix_order:
                for j in matrix_order:
                    distances[temp_dict[i]][temp_dict[j]] = data_travel_time[route_ids[route_number]][i][j]
            
            self.tt_matrices.append(distances)

    def get_stops_tt_matrices(self):
        return (self.stops, self.tt_matrices, self.travel_times_dict)