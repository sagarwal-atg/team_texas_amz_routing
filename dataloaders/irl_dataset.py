import os
import time
from collections import namedtuple
from functools import total_ordering
from typing import Any, List

import numpy as np
import torch
from IPython import embed
from nptyping import NDArray
from numpy import concatenate as concat
from sklearn import preprocessing
from torch.utils import data
from torch.utils.data import Dataset

from .data import (
    SCORE,
    PackageData,
    RouteData,
    RouteDatum,
    SequenceData,
    TravelTimeData,
    TravelTimeDatum,
)
from .utils import ENDC, OKGREEN

IntMatrix = NDArray[(Any, Any), np.int32]
FloatMatrix = NDArray[(Any, Any), np.float]
BoolMatrix = NDArray[(Any, Any), np.bool]

IRLData = namedtuple(
    "IRLData",
    [
        "travel_times",
        "link_features",
        "route_features",
        "time_constraints",
        "stop_ids",
        "travel_time_dict",
        "label",
        "binary_mat",
    ],
)

###################################
# Feature Extractors - Begin
###################################
def extract_travel_times(
    travel_times: TravelTimeDatum, stop_ids: List[str]
) -> FloatMatrix:
    """
    stop_ids: a list which is in the same order as the labels
        ex: ['A', 'B', 'C']
    Returns: a matrix whose ij entry is the travel time from stop_ids[i] -> stop_ids[j]
        The times are normalized by dividing across the rows so that the sum time from
        stop_ids[i] -> all other stops is 1.
    """
    times = travel_times.as_matrix(stop_ids)
    # times /= times.sum(axis=1)[:,np.newaxis]
    return times


def extract_zone_crossings(route_data: RouteDatum, stop_ids: List[str]) -> BoolMatrix:
    """
    Returns: A matrix whose ij entry is 1 if there is a zone crossing between
        stop stop_ids[i] and stop_ids[j] (and 0 otherwise)
    """
    # get a map of zone id's for each stop
    stop_zones = route_data.get_zones(stop_ids)

    # get a matrix of zone crossings (note: it should be symmetric)
    zones = list(stop_zones.values())
    x, y = np.unique(np.array(zones), return_inverse=True)
    zone_crossings = y[:, None] != y
    # zone_crossing = lambda stop1, stop2: stop_zones[stop1] != stop_zones[stop2]
    # zone_crossings = np.zeros((len(stop_ids), len(stop_ids)))
    # for i, stop1 in enumerate(stop_ids):
    #     for j, stop2 in enumerate(stop_ids):
    #         zone_crossings[i, j] = zone_crossing(stop1, stop2)
    return zone_crossings


###################################
# Feature Extractors - End
###################################


def right_pad2d(m, width, constant=0):
    assert width >= m.shape[1]
    m2 = np.ones((m.shape[0], width)) * constant
    m2[:, : m.shape[1]] = m
    return m2


def get_x_matrix(
    route_features, link_features, route_lengths, max_route_len
) -> FloatMatrix:
    """
    Returns a matrix where the ith row is the features FROM a stop in a route and the
        columns j to j+n are the n features TO the jth stop
        (where n = num_link_features + num_route_features)
    """
    # shape(route_data)=[n_routes, num_route_features]
    route_data = np.repeat(route_features, route_lengths, axis=0)
    # shape(route_date)=[sum(route_lengths), num_route_features]
    route_data = np.tile(route_data[np.newaxis], [1, 1, max_route_len])
    # shape(route_data)=[num_route_features, sum(route_lengths), max_route_len]
    x = concat([link_features])  # skipping route_data for now
    # shape(x)=[num_features, sum(route_lengths), max_route_len]
    x = np.transpose(x, (1, 2, 0))
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    # shape(x)=[sum(route_lengths), max_route_len * num_features]
    return x


class ClassificationDataset(Dataset):
    def __init__(self, data_config):
        route_data = RouteData.from_file(data_config.route_path)
        sequence_data = SequenceData.from_file(data_config.sequence_path)
        travel_time_data = TravelTimeData.from_file(data_config.travel_time_path)

        route_ids = route_data.get_high_score_ids()
        route_ids = route_ids[slice(data_config.slice_begin, data_config.slice_end)]
        self.route_ids = route_ids

        print(f"Using data from {len(route_ids)} routes.")
        route_lengths = [len(sequence_data[route]) for route in route_ids]
        self.route_lengths = route_lengths
        self.max_route_len = max(route_lengths)

        def get_route_features(route_id):
            veh_cap = route_data[route_id]._data.executor_capacity_cm3
            # add any other functions here for more route features
            return np.array([veh_cap]).reshape(
                1, 1
            )  # remove reshape once added more features

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
        link_features = concat(
            [get_link_features(route_id) for route_id in route_ids], axis=1
        )
        route_features = concat(
            [get_route_features(route_id) for route_id in route_ids]
        )

        # Extract labels
        lables = concat(
            [sequence_data[route_id].get_route_order() for route_id in route_ids]
        )

        x = get_x_matrix(
            route_features, link_features, route_lengths, self.max_route_len
        )

        self.num_features = (
            x.shape[1] // self.max_route_len
        )  # is same as link_features.shape[0] + route_features.shape[1]
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(lables)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class IRLDataset(Dataset):
    def __init__(self, data_config):
        route_data = RouteData.from_file(data_config.route_path)
        sequence_data = SequenceData.from_file(data_config.sequence_path)
        travel_time_data = TravelTimeData.from_file(data_config.travel_time_path)
        package_data = PackageData.from_file(data_config.package_path)

        route_score = SCORE.HIGH
        if data_config.route_score == "medium":
            route_score = SCORE.MEDIUM
        if data_config.route_score == "low":
            route_score = SCORE.LOW
        print(route_score)

        route_ids = route_data.get_routes_with_score_ids(route_score)

        route_ids = route_ids[slice(data_config.slice_begin, data_config.slice_end)]
        self.route_ids = route_ids

        print(f"Using data from {len(route_ids)} routes.")
        route_lengths = [len(sequence_data[route]) for route in route_ids]
        self.route_lengths = route_lengths
        self.max_route_len = max(route_lengths)

        def get_route_features(route_id):
            veh_cap = route_data[route_id]._data.executor_capacity_cm3
            # add any other functions here for more route features
            return np.array([veh_cap]).reshape(
                1, 1
            )  # remove reshape once added more features

        def get_link_features(route_id):
            """
            Returns: a matrix of shape [n, route_len, route_len] where n is the number of features
            """
            stop_ids = sequence_data[route_id].get_stop_ids()

            # add service time to travel time matrix
            # times = extract_travel_times(travel_time_data[route_id], stop_ids)
            # ser_times = package_data.find_service_times(stop_ids)
            # times += ser_times
            # times = right_pad2d(times, self.max_route_len, 0)

            zone_crossings = extract_zone_crossings(route_data[route_id], stop_ids)
            # zone_crossings = right_pad2d(zone_crossings, self.max_route_len, 1)

            # add any other functions here for more link features.
            return np.array([zone_crossings])

        def get_travel_time(route_id):
            """
            Returns: a matrix of shape [route_len, route_len]
            """
            stop_ids = sequence_data[route_id].get_stop_ids()

            # add service time to travel time matrix
            times = extract_travel_times(travel_time_data[route_id], stop_ids)
            ser_times = package_data[route_id].find_service_times(stop_ids)
            times += ser_times

            return times

        def get_time_constraints(route_id):
            """
            Returns: a list of tuples [start, end]. Start and end time of the constraint. None if passed no constraints.
            """
            stop_ids = sequence_data[route_id].get_stop_ids()
            route_start = route_data[route_id].get_start_time()
            return package_data[route_id].find_time_windows(route_start, stop_ids)

        def get_scoring_function_inputs(route_id):
            """
            Returns: stop ids and travel time dict
            """
            stop_ids = sequence_data[route_id].get_stop_ids()
            travel_time_dict = travel_time_data[route_id]._data
            return stop_ids, travel_time_dict

        self.x = []

        for route_id in route_ids:
            travel_times = get_travel_time(route_id)
            link_features = get_link_features(route_id)
            route_features = get_route_features(route_id)
            time_constraints = get_time_constraints(route_id)
            label = sequence_data[route_id].get_sorted_route_by_index()
            stop_ids, travel_time_dict = get_scoring_function_inputs(route_id)
            self.x.append(
                (
                    travel_times,
                    link_features,
                    route_features,
                    time_constraints,
                    stop_ids,
                    travel_time_dict,
                    label,
                )
            )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


def seq_binary_mat(seq):
    route_len = len(seq)
    binary_mat = np.zeros(((route_len), route_len))
    for idx, stop in enumerate(seq):
        if (idx + 1) == route_len:
            binary_mat[seq[idx], seq[0]] = 1
        else:
            binary_mat[seq[idx], seq[idx + 1]] = 1
    return binary_mat


def irl_nn_collate(batch):
    nn_data = [item[0] for item in batch]
    other_data = [item[1] for item in batch]
    scaled_tc_data = [item[2] for item in batch]
    return [nn_data, other_data, scaled_tc_data]


def find_closest_idx(travel_times, num_closest):
    assert num_closest % 2 == 0, "Only even num closest is supported"
    closest_idx = np.zeros((travel_times.shape[0], travel_times.shape[1], num_closest))

    for idx in range(travel_times.shape[0]):
        for jdx in range(travel_times.shape[1]):
            top_n_rows = np.argsort(travel_times[:, jdx])[: int(num_closest / 2)]
            top_n_cols = np.argsort(travel_times[idx, :])[: int(num_closest / 2)]
            closest_idx[idx, jdx, :] = np.array([top_n_rows, top_n_cols]).reshape(
                (num_closest,)
            )
    return closest_idx


class IRLNNDataset(Dataset):
    def __init__(self, data_config, cache_path=None):

        start_time = time.time()

        route_data = RouteData.from_file(data_config.route_path)
        sequence_data = SequenceData.from_file(data_config.sequence_path)
        travel_time_data = TravelTimeData.from_file(data_config.travel_time_path)
        package_data = PackageData.from_file(data_config.package_path)

        route_score = SCORE.HIGH
        if data_config.route_score == "medium":
            route_score = SCORE.MEDIUM
        if data_config.route_score == "low":
            route_score = SCORE.LOW

        route_ids = route_data.get_routes_with_score_ids(route_score)
        station_code_dict = route_data.make_station_code_indxes()

        route_ids = route_ids[slice(data_config.slice_begin, data_config.slice_end)]
        self.route_ids = route_ids

        route_lengths = [len(sequence_data[route]) for route in route_ids]
        self.route_lengths = route_lengths
        self.max_route_len = max(route_lengths)

        def get_route_features(route_id):
            veh_cap = route_data[route_id]._data.executor_capacity_cm3
            station_code = station_code_dict[route_data[route_id].get_station_code()]

            return np.array([veh_cap, station_code])

        def get_link_features(route_id):
            """
            Returns: a matrix of shape [n, route_len, route_len] where n is the number of features
            """
            stop_ids = sequence_data[route_id].get_stop_ids()

            # add service time to travel time matrix
            # times = extract_travel_times(travel_time_data[route_id], stop_ids)
            # ser_times = package_data.find_service_times(stop_ids)
            # times += ser_times
            # times = right_pad2d(times, self.max_route_len, 0)

            zone_crossings = extract_zone_crossings(route_data[route_id], stop_ids)
            # zone_crossings = right_pad2d(zone_crossings, self.max_route_len, 1)
            geo_dist_mat = route_data[route_id].get_geo_dist_mat(stop_ids)

            # add any other functions here for more link features.
            return np.array([zone_crossings, geo_dist_mat])

        def get_travel_time(route_id):
            """
            Returns: a matrix of shape [route_len, route_len]
            """
            stop_ids = sequence_data[route_id].get_stop_ids()

            # add service time to travel time matrix
            times = extract_travel_times(travel_time_data[route_id], stop_ids)
            ser_times = package_data[route_id].find_service_times(stop_ids)
            times += ser_times

            return times

        def get_time_constraints(route_id):
            """
            Returns: a list of tuples [start, end]. Start and end time of the constraint. None if passed no constraints.
            """
            stop_ids = sequence_data[route_id].get_stop_ids()
            route_start = route_data[route_id].get_start_time()
            return package_data[route_id].find_time_windows(route_start, stop_ids)

        def get_scoring_function_inputs(route_id):
            """
            Returns: stop ids and travel time dict
            """
            stop_ids = sequence_data[route_id].get_stop_ids()
            travel_time_dict = travel_time_data[route_id]._data
            return stop_ids, travel_time_dict

        self.x = []
        closest_idxs_for_route = []

        for route_id in route_ids:
            travel_times = get_travel_time(route_id)
            link_features = get_link_features(route_id)
            route_features = get_route_features(route_id)
            time_constraints = get_time_constraints(route_id)
            label = sequence_data[route_id].get_sorted_route_by_index()
            stop_ids, travel_time_dict = get_scoring_function_inputs(route_id)
            binary_mat = seq_binary_mat(label)

            irl_data = IRLData(
                travel_times=travel_times,
                link_features=link_features,
                route_features=route_features,
                time_constraints=time_constraints,
                stop_ids=stop_ids,
                travel_time_dict=travel_time_dict,
                label=label,
                binary_mat=binary_mat,
            )

            if data_config.num_neighbors > 1:
                closest_idxs_for_route.append(
                    find_closest_idx(travel_times, data_config.num_neighbors)
                )

            self.x.append(irl_data)

        cache_file_path = os.path.join(cache_path, "cache_path.npz")
        print(cache_file_path, os.path.isfile(cache_file_path))
        if os.path.isfile(cache_file_path) and data_config.use_cache:
            data = np.load(cache_file_path, allow_pickle=True)
            self.nn_data = data["arr_0"]
            self.scaled_tc_data = data["arr_1"]
            print(
                OKGREEN
                + " Using Cached Data !!. Deleted file or change flag "
                + "if you have made new changes to the feature generation"
                + ENDC
            )
        else:
            self.nn_data, self.scaled_tc_data = self.preprocess(data_config)
            if len(closest_idxs_for_route) != 0:
                self.nn_data = self.add_neighbors(
                    self.nn_data, data_config.num_neighbors, closest_idxs_for_route
                )
            np.savez(cache_file_path, self.nn_data, self.scaled_tc_data)

        print(
            f"Using data from {len(self.x)} routes in {time.time() - start_time} secs"
        )

    def preprocess(self, data_config):
        num_routes = len(self.x)
        num_link_features = data_config.num_link_features
        num_route_features = data_config.num_route_features

        travel_times = [None] * num_routes
        link_features = [None] * num_routes
        route_features = np.zeros((num_routes, num_route_features))
        time_constraints = [None] * num_routes

        transformed_data = [None] * num_routes

        self.total_num_links = 0
        for idx, data in enumerate(self.x):
            tt = data.travel_times
            assert len(tt.shape) == 2
            route_len = tt.shape[0]

            travel_times[idx] = tt
            self.total_num_links += tt.shape[0] * tt.shape[0]

            lf = data.link_features

            assert (
                lf.shape[0] == num_link_features
            ), "Fix the size of link features array function if this is changed."

            link_features[idx] = lf

            rf = data.route_features
            assert (
                rf.shape[0] == num_route_features
            ), "Fix the size of route features array function if this is changed."

            route_features[idx] = rf

            time_constraints[idx] = np.array(data.time_constraints)

        tt_np = np.zeros((1, self.total_num_links))
        idx_so_far = 0
        for tt_data in travel_times:
            flat_tt = tt_data.flatten()
            tt_np[:, idx_so_far : (idx_so_far + flat_tt.shape[0])] = flat_tt
            idx_so_far += flat_tt.shape[0]

        # Time Constraint Scaling
        tc_np = np.zeros((sum([a.shape[0] for a in time_constraints]), 2))
        # idx_so_far = 0
        # for tc_data in time_constraints:
        #     num_stops = tc_data.shape[0]
        #     tc_np[idx_so_far:(idx_so_far + num_stops), :] = tc_data
        #     idx_so_far += num_stops

        # tt_scaler = preprocessing.StandardScaler(with_mean=False).fit(tt_np.T)
        # tt_np = tt_scaler.transform(tt_np.T)
        # tt_np = tt_np.T

        # tc_np = (tc_np) / (tt_scaler.var_)
        # tc_np = tc_np.astype(np.int)

        # Time Constraint Scaling
        scaled_tc_data = []
        idx_so_far = 0
        for tc_data in time_constraints:
            num_stops = tc_data.shape[0]
            scaled_tc_data.append(tc_np[idx_so_far : (idx_so_far + num_stops)].tolist())
            idx_so_far += num_stops

        rf_scaler = preprocessing.StandardScaler().fit(route_features)
        route_features = rf_scaler.transform(route_features)

        idx_so_far = 0
        for idx, data in enumerate(self.x):
            route_len = data.travel_times.shape[0]

            nn_data_np = np.zeros(
                (route_len * route_len, num_link_features + num_route_features + 1)
            )
            nn_data_np[:, 0] = tt_np[
                0, idx_so_far : (idx_so_far + (route_len * route_len))
            ]
            idx_so_far += route_len * route_len

            nn_data_np[:, 1:-num_route_features] = link_features[idx].T.reshape(
                (route_len * route_len, num_link_features)
            )

            nn_data_np[:, -num_route_features:] = (
                np.ones((route_len * route_len, num_route_features))
                * route_features[idx]
            )
            transformed_data[idx] = nn_data_np

        return transformed_data, scaled_tc_data

    def add_neighbors(self, nn_data, num_neighbors, closest_idxs_for_route):
        new_nn_data = []
        num_features = nn_data[0].shape[1]

        for d_idx in range(len(closest_idxs_for_route)):
            route_len, route_len, _ = closest_idxs_for_route[d_idx].shape
            num_links_in_route = route_len * route_len
            new_nn_data_idx = np.zeros(
                (num_links_in_route, (num_features) * (num_neighbors + 1))
            )

            new_nn_data_idx[:, :num_features] = nn_data[d_idx]

            nn_data_mat = nn_data[d_idx].reshape((route_len, route_len, num_features))

            for idx in range(route_len):
                for jdx in range(route_len):
                    for nidx in range(num_neighbors):
                        if nidx < num_neighbors / 2:
                            new_nn_data_idx[
                                jdx * route_len + idx,
                                (nidx + 1) * num_features : (nidx + 2) * num_features,
                            ] = nn_data_mat[
                                int(closest_idxs_for_route[d_idx][idx, jdx, nidx]),
                                jdx,
                                :,
                            ]
                        else:
                            new_nn_data_idx[
                                jdx * route_len + idx,
                                (nidx + 1) * num_features : (nidx + 2) * num_features,
                            ] = nn_data_mat[
                                idx,
                                int(closest_idxs_for_route[d_idx][idx, jdx, nidx]),
                                :,
                            ]
            new_nn_data.append(new_nn_data_idx)

        return new_nn_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        nn_data = torch.from_numpy(self.nn_data[idx]).type(torch.FloatTensor)
        other_data = self.x[idx]
        scaled_tc_data = self.scaled_tc_data[idx]
        return [nn_data, other_data, scaled_tc_data]
