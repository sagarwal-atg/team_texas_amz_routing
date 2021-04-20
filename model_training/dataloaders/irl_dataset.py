from typing import List, Any
import numpy as np
from numpy import concatenate as concat
from nptyping import NDArray
import torch
from torch.utils.data import Dataset

from .data import RouteData, SequenceData, TravelTimeData, RouteData, RouteDatum, TravelTimeDatum

IntMatrix = NDArray[(Any, Any), np.int32]
FloatMatrix = NDArray[(Any, Any), np.float]
BoolMatrix = NDArray[(Any, Any), np.bool]


###################################
# Feature Extractors - Begin 
###################################
def extract_travel_times(travel_times: TravelTimeDatum, stop_ids: List[str]) -> FloatMatrix:
    """
    stop_ids: a list which is in the same order as the labels
        ex: ['A', 'B', 'C']
    Returns: a matrix whose ij entry is the travel time from stop_ids[i] -> stop_ids[j]
        The times are normalized by dividing across the rows so that the sum time from
        stop_ids[i] -> all other stops is 1.
    """
    times = travel_times.as_matrix(stop_ids)
    times /= times.sum(axis=1)[:,np.newaxis]
    return times

def extract_zone_crossings(route_data: RouteDatum, stop_ids: List[str]) -> BoolMatrix:
    """
    Returns: A matrix whose ij entry is 1 if there is a zone crossing between
        stop stop_ids[i] and stop_ids[j] (and 0 otherwise)
    """
    # get a map of zone id's for each stop
    stop_zones = route_data.get_zones(stop_ids)
    zone_crossing = lambda stop1, stop2: stop_zones[stop1] != stop_zones[stop2]
    # get a matrix of zone crossings (note: it should be symmetric)
    zone_crossings = np.zeros((len(stop_ids), len(stop_ids)))
    for i, stop1 in enumerate(stop_ids):
        for j, stop2 in enumerate(stop_ids):
            zone_crossings[i, j] = zone_crossing(stop1, stop2)
    return zone_crossings

###################################
# Feature Extractors - End 
###################################


def right_pad2d(m, width, constant=0):
    assert width >= m.shape[1]
    m2 = np.ones((m.shape[0], width)) * constant
    m2[:,:m.shape[1]] = m
    return m2

def get_x_matrix(route_features, link_features, route_lengths, max_route_len) -> FloatMatrix:
    """
        Returns a matrix where the ith row is the features FROM a stop in a route and the 
            columns j to j+n are the n features TO the jth stop
            (where n = num_link_features + num_route_features)
    """
    # shape(route_data)=[n_routes, num_route_features]
    route_data = np.repeat(route_features, route_lengths, axis=0)
    # shape(route_date)=[sum(route_lengths), num_route_features]
    route_data = np.tile(route_data[np.newaxis], [1,1,max_route_len])
    # shape(route_data)=[num_route_features, sum(route_lengths), max_route_len]
    x = concat([link_features]) # skipping route_data for now
    # shape(x)=[num_features, sum(route_lengths), max_route_len]
    x = np.transpose(x, (1,2,0))
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    # shape(x)=[sum(route_lengths), max_route_len * num_features]
    return x

class IRLDataset(Dataset):
    def __init__(self, config, slice_begin=None, slice_end=None):
        route_data = RouteData.from_file(config.route_path)
        sequence_data = SequenceData.from_file(config.sequence_path)
        travel_time_data = TravelTimeData.from_file(config.travel_time_path)

        route_ids = route_data.get_high_score_ids()
        route_ids = route_ids[slice(slice_begin, slice_end)]
        print(f'Using data from {len(route_ids)} routes.')
        route_lengths = [len(sequence_data[route]) for route in route_ids]
        self.max_route_len = max(route_lengths)

        def get_route_features(route_id):
            veh_cap = route_data[route_id]._data.executor_capacity_cm3
            # add any other functions here for more route features
            return np.array([veh_cap]).reshape(1, 1) # remove reshape once added more features

        def get_link_features(route_id):
            """
            Returns: a matrix of shape [route_len, max_route_len, n] where n is the number of features
            """
            stop_ids = sequence_data[route_id].get_stop_ids()

            times = extract_travel_times(travel_time_data[route_id], stop_ids)
            times = right_pad2d(times, self.max_route_len, 0)

            zone_crossings = extract_zone_crossings(route_data[route_id], stop_ids)
            zone_crossings = right_pad2d(zone_crossings, self.max_route_len, 1)

            # add any other functions here for more link features.
            return np.array([times, zone_crossings])

        # Extract features
        link_features = concat([
            get_link_features(route_id) for route_id in route_ids], axis=1)
        route_features = concat([
            get_route_features(route_id) for route_id in route_ids])

        # Extract labels
        lables = concat([
            sequence_data[route_id].get_route_order() for route_id in route_ids])

        x = get_x_matrix(route_features, link_features, route_lengths, self.max_route_len)

        self.num_features = x.shape[1] // self.max_route_len # is same as link_features.shape[0] + route_features.shape[1]
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(lables)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
