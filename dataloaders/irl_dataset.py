import json
from typing import Dict, List, Sequence
import numpy as np
from numpy import concatenate as concat
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
import enum

import torch
from torch.utils.data import Dataset


def order_stops(stop_sequence: Dict[str, int]) -> List[str]:
    """
    stop_sequence: a dict whose values indicate the order we visited the stops
        ex: {'A': 0, 'B': 2, 'C': 1}
    Returns: a list ordered by the values in stop_sequence
        ex: ['A', 'C', 'B']
    """
    return sorted(stop_sequence, key=lambda k: stop_sequence[k])

def get_route_order(stop_sequence: Dict[str, int]):
    """
    stop_sequence: a dict whose values indicate the order we visited the stops
        ex: {'A': 0, 'B': 2, 'C': 1}
    Returns: a list of ints whose index is the FROM index in stop_sequence and whose value is
        the TO index in stop_sequence.
        ex: [2, 0, 1] (meaning that we go from [A->C, B->A, C->B])
    """
    stops_ordered = order_stops(stop_sequence)
    next_stop = {stop: next_stop for stop, next_stop in zip(np.roll(stops_ordered, 1), stops_ordered)}
    stop_ids = list(stop_sequence.keys())
    stop_idx = {k: i for i, k in enumerate(stop_ids)}
    labels = [stop_idx[next_stop[stop]] for stop in stop_idx.keys()] 
    return labels

def extract_travel_times(stop_ids: List[str], travel_times: List[List[float]]):
    """
    stop_ids: a list which is in the same order as the labels
        ex: ['A', 'B', 'C']
    Returns: a matrix whose ij entry is the travel time from stop_ids[i] -> stop_ids[j]
        The times are normalized by dividing across the rows so that the sum time from
        stop_ids[i] -> all other stops is 1.
    """
    stop_idx = {k: i for i, k in enumerate(stop_ids)}
    route_len = len(stop_ids)
    times = np.zeros((route_len, route_len))
    for i in stop_ids:
        for j in stop_ids:
            times[stop_idx[i]][stop_idx[j]] = travel_times[i][j]
    times /= times.sum(axis=1)[:,np.newaxis]

    return times

def right_pad(m, width, constant=0):
    assert width >= m.shape[2]
    m2 = np.ones((m.shape[0], m.shape[1], width)) * constant
    m2[:,:,:m.shape[-1]] = m
    return m2

class IRLDataset(Dataset):
    def __init__(self, paths, slice_begin=None, slice_end=None):
        with open(paths.route) as f:
            route_data = json.load(f)

        with open(paths.labels) as f:
            sequence_data = json.load(f)

        with open(paths.travel_times) as f:
            travel_time_data = json.load(f)

        # with open(paths.packages) as f:
        #     package_data = json.load(f)

        route_ids = [k for k, v in route_data.items() if v['route_score'] == 'High']
        route_ids = route_ids[slice(slice_begin, slice_end)]
        n_routes = len(route_ids)
        print(f'Using data from {n_routes} routes.')
        route_lengths = [len(sequence_data[route_id]['actual']) for route_id in route_ids]

        self.max_route_len = max(route_lengths)

        def get_route_features(route_id):
            vehicle_cap = route_data[route_id]['executor_capacity_cm3']
            # add any other functions here for more route features
            return np.array([vehicle_cap])

        def get_link_features(route_id):
            """
            Returns: a matrix of shape [route_len, route_len, n] where n is the number of features
            """
            stop_ids = list(sequence_data[route_id]['actual'].keys())
            times = extract_travel_times(stop_ids, travel_time_data[route_id])
            # add any other functions here for more link features
            return np.array([times])

        # Extract features
        link_features = [get_link_features(route_id) for route_id in route_ids]
        route_features = concat([
            get_route_features(route_id) for route_id in route_ids])

        # Make all matrices the same width by adding 1's to the right side
        link_features = concat([
            right_pad(matrix, self.max_route_len) for matrix in link_features], axis=1)

        # Extract labels
        lables = concat([
            get_route_order(sequence_data[route_id]['actual']) for route_id in route_ids])

        # shape(route_date)=[n_routes, num_route_features]
        route_data = np.repeat(route_features, route_lengths, axis=0)
        # shape(route_date)=[sum(route_lengths), num_route_features]
        route_data = np.tile(route_data[np.newaxis,:,np.newaxis], [1,1,self.max_route_len])
        # shape(route_date)=[max_route_len, sum(route_lengths), num_route_features]
        x = concat([link_features]) # skipping route_data for now
        self.num_features = x.shape[0]
        # shape(x)=[num_features, sum(route_lengths), max_route_len]
        x = np.transpose(x, (1,2,0))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        # shape(x)=[sum(route_lengths), max_route_len * num_features]

        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(lables)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
