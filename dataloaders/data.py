import json
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Union

import numpy as np
from haversine import Unit, haversine
from munch import Munch  # used to give dot accessing to dict
from sklearn.preprocessing import OneHotEncoder


class RouteDatum:
    def __init__(self, data):
        self._data = Munch(data)
        self.zone_d = self.get_zones(self.get_stop_ids())

    def get_stop_ids(self):
        return list(self._data["stops"].keys())

    def get_depot(self) -> str:
        stops = self.get_stops()
        for idx, key in enumerate(stops):
            if stops[key]["type"] != "Dropoff":
                return key, idx

    def get_stops(self, stop_ids: List[str] = None):
        stops = self._data.stops
        if stop_ids:
            stops = {key: stops[key] for key in stop_ids}
        return stops

    def get_zone_mat(self, max_num_zones):

        zone_d = self.zone_d
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        zone_list = list(zone_d.values())
        ohc_zone = enc.fit_transform(np.array(zone_list).reshape(len(zone_list), 1))
        num_zones = enc.categories_[0].shape[0]

        zone_mat = [[None] * len(zone_list)] * len(zone_list)
        for idx, _ in enumerate(zone_d):
            for jdx, _ in enumerate(zone_d):
                zone_np = np.zeros((max_num_zones,))
                # zone_np[:num_zones] = enc.transform(
                #     np.array(zone_d[stop_a]).reshape(1, 1)
                # ) + enc.transform(np.array(zone_d[stop_b]).reshape(1, 1))
                zone_np[:num_zones] = ohc_zone[idx] + ohc_zone[jdx]
                zone_mat[idx][jdx] = zone_np
        return np.array(zone_mat)

    def get_zone_ohc(self):

        zone_d = self.zone_d
        # enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        zone_list = list(zone_d.values())
        # ohc_zones = enc.fit_transform(np.array(zone_list).reshape(len(zone_list), 1))
        # num_zones = enc.categories_[0].shape[0]

        # zone_mat = [None] * len(zone_list)
        # for idx, _ in enumerate(zone_d):
        #     zone_mat[idx] = ohc_zones[idx]

        # ret_np = np.array(zone_mat)
        ret_np = np.array(zone_list).reshape(len(zone_list), 1)
        return ret_np.T

    # def get_stop_zones_ohc(self):

    #     zone_d = self.zone_d
    #     enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    #     zone_list = list(zone_d.values())
    #     ohc_zones = enc.fit_transform(np.array(zone_list).reshape(len(zone_list), 1))
    #     num_zones = enc.categories_[0].shape[0]

    #     zone_mat = [None] * len(zone_list)
    #     for idx, _ in enumerate(zone_d):
    #         zone_mat[idx] = ohc_zones[idx]

    #     ret_np = np.array(zone_mat)
    #     return ret_np.T

    def get_zones(self, stop_ids: List[str] = None) -> Dict[str, str]:
        """
        Returns: a map of zone id's for each stop
            ex: {'A': 'zone1', 'B': 'zone2', 'C': 'zone3'}
            if zone_id is nan, set zone_id to be 'zone_' + stop_id
        """
        zone_d = {}
        for key, stop in self.get_stops(stop_ids).items():
            zone_d[key] = (
                stop["zone_id"] if isinstance(stop["zone_id"], str) else "zone_" + key
            )
        return zone_d

    def get_score(self):
        return self._data.route_score

    def is_score(self, score):
        return self.get_score() == score

    def get_start_time(self):
        return datetime.fromisoformat(
            self._data.date_YYYY_MM_DD + " " + self._data.departure_time_utc
        )

    def get_station_code(self):
        return self._data.station_code

    def get_geo_dist(self, stop_a, stop_b):
        stop_a_dict = self._data.stops[stop_a]
        stop_b_dict = self._data.stops[stop_b]

        stop_a_ = (stop_a_dict["lat"], stop_a_dict["lng"])
        stop_b_ = (stop_b_dict["lat"], stop_b_dict["lng"])

        return haversine(stop_a_, stop_b_)

    def get_lat_long(self, stop_ids):
        lat_long = np.zeros((2, len(stop_ids)))
        for adx, stop_id_a in enumerate(stop_ids):
            stop_a_dict = self._data.stops[stop_id_a]
            lat_long[0, adx] = stop_a_dict["lat"]
            lat_long[1, adx] = stop_a_dict["lng"]
        return lat_long

    def get_geo_dist_mat(self, stop_ids):
        geo_dist_mat = np.zeros((len(stop_ids), len(stop_ids)))
        for adx, stop_id_a in enumerate(stop_ids):
            for bdx, stop_id_b in enumerate(stop_ids):
                # TODO small optimization: use the same dist for ab and ba
                geo_dist_mat[adx, bdx] = self.get_geo_dist(stop_id_a, stop_id_b)

        return geo_dist_mat

    def get_depot_distance_(self, stop_ids):
        depot_dist_ = np.zeros((len(stop_ids)))
        depot, _ = self.get_depot()
        for adx, stop_id_a in enumerate(stop_ids):
            depot_dist_[adx] = self.get_geo_dist(depot, stop_id_a)
        return depot_dist_


class SequenceDatum:
    """
    Data about the actual route taken by the driver
    """

    def __init__(self, data: Dict[str, Dict[str, int]]):
        """
        data: a dict whose values indicate the order we visited the stops
            ex: {'actual': {'A': 0, 'B': 2, 'C': 1}}
        """
        # self.tt_dicts = data['actual']
        self._data = data
        self._stops = data["actual"]
        # sorted_data = dict(sorted(self.tt_dicts.items(), key=lambda item: item[1]))
        # self._stops = sorted_data

    def get_stop_ids(self) -> List[str]:
        """
        Returns a list of stops in the order they come in the data
            ex: ['A', 'B', 'C']
        """
        return list(self._stops.keys())

    def _get_actual_order(self) -> List[str]:
        """
        Returns: a list ordered by the values in stop_sequence
            ex: ['A', 'C', 'B']
        """
        return sorted(self._stops, key=lambda k: self._stops[k])

    def get_sorted_route_by_index(self):
        """ """
        labels = np.argsort(list(self._stops.values()))
        # labels = np.arange(len(self._stops))
        return labels

    def get_route_order(self):
        """
        stop_sequence: a dict whose values indicate the order we visited the stops
            ex: {'A': 0, 'B': 2, 'C': 1}
        Returns: a list of ints whose index is the FROM index in stop_sequence and whose value is
            the TO index in stop_sequence.
            ex: [2, 0, 1] (meaning that we go from [A->C, B->A, C->B])
        """
        stops_ordered = self._get_actual_order()
        next_stop = {
            stop: next_stop
            for stop, next_stop in zip(np.roll(stops_ordered, 1), stops_ordered)
        }
        stop_idx = {k: i for i, k in enumerate(self._stops)}
        labels = [stop_idx[next_stop[stop]] for stop in stop_idx.keys()]
        return labels

    def __len__(self):
        return len(self.get_stop_ids())


class TravelTimeDatum:
    def __init__(self, data: Dict[str, Dict[str, float]]) -> None:
        self._data = data

    def as_matrix(self, stop_ids: List[str] = None):
        route_len = len(stop_ids)
        times = np.zeros((route_len, route_len))
        for i, stop1 in enumerate(stop_ids):
            for j, stop2 in enumerate(stop_ids):
                times[i][j] = self._data[stop1][stop2]
        return times


class PackageDatum:
    def __init__(self, data) -> None:
        self._data = data

    def find_service_times(self, stop_ids):
        time_windows = []
        for stop_id in stop_ids:
            time_windows.append(self.find_service_time_for_stop(stop_id))
        return time_windows

    def find_service_time_for_stop(self, stop_id):
        packages = self._data[stop_id]
        t = 0
        for p, pinfo in packages.items():
            t = t + pinfo["planned_service_time_seconds"]
        return t

    def find_time_windows(self, route_start, stop_ids):
        time_windows = []
        for stop_id in stop_ids:
            time_windows.append(self.find_time_window_for_stop(route_start, stop_id))
        return time_windows

    def find_time_window_for_stop(self, route_start, stop_id):
        """Inputs: Key: route_id
        Stop_id: 2 alphabet id for stop"""

        packages = self._data[stop_id]
        end_stop = datetime.max
        start_stop = route_start
        delta_max = end_stop - start_stop
        for pid, pinfo in packages.items():
            if isinstance(pinfo["time_window"]["end_time_utc"], str):
                end = datetime.fromisoformat(pinfo["time_window"]["end_time_utc"])
                start = datetime.fromisoformat(pinfo["time_window"]["start_time_utc"])
                end_stop = min(end, end_stop)
                start_stop = max(start, start_stop)
        delta = end_stop - start_stop
        if delta < delta_max:
            return (
                int((start_stop - route_start).total_seconds()),
                int((end_stop - route_start).total_seconds()),
            )
        else:
            return (0, 57600)

    def get_package_info(self):

        """route_id example: RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77
        data_package: the dictionary retrieved from package_data.json"""

        num_package_list = []
        total_service_time_list = []
        largest_package_volume_list = []
        avg_volume_of_package_list = []

        for stop in self._data:
            temp_dict = self._data[stop]

            if len(temp_dict) == 0:
                num_package_list.append(0)
                total_service_time_list.append(0)
                largest_package_volume_list.append(0)
                avg_volume_of_package_list.append(0)

            else:
                num_package = len(temp_dict)

                service_time_total = 0
                max_package_volume = 0
                total_package_volume = 0

                for package in temp_dict:

                    service_time_total = (
                        service_time_total
                        + temp_dict[package]["planned_service_time_seconds"]
                    )
                    current_volume = (
                        temp_dict[package]["dimensions"]["depth_cm"]
                        * temp_dict[package]["dimensions"]["height_cm"]
                        * temp_dict[package]["dimensions"]["width_cm"]
                    )
                    if current_volume > max_package_volume:
                        max_package_volume = current_volume
                    total_package_volume = total_package_volume + current_volume

                num_package_list.append(num_package)
                total_service_time_list.append(service_time_total)
                largest_package_volume_list.append(max_package_volume)
                avg_volume_of_package_list.append(total_package_volume / num_package)

        my_dict = {}

        my_dict["num_package_dest"] = np.repeat(
            [num_package_list], len(num_package_list), axis=0
        )
        my_dict["num_package_source"] = np.repeat(
            [num_package_list], len(num_package_list), axis=0
        ).T

        my_dict["total_service_time_dest"] = np.repeat(
            [total_service_time_list], len(num_package_list), axis=0
        )
        my_dict["total_service_time_source"] = np.repeat(
            [total_service_time_list], len(num_package_list), axis=0
        ).T

        my_dict["largest_package_volume_dest"] = np.repeat(
            [largest_package_volume_list], len(num_package_list), axis=0
        )
        my_dict["largest_package_volume_source"] = np.repeat(
            [largest_package_volume_list], len(num_package_list), axis=0
        ).T

        my_dict["avg_volume_of_package_dest"] = np.repeat(
            [avg_volume_of_package_list], len(num_package_list), axis=0
        )
        my_dict["avg_volume_of_package_source"] = np.repeat(
            [avg_volume_of_package_list], len(num_package_list), axis=0
        ).T

        return my_dict


class AmazonData:
    def __init__(self, data, constructor: Union[RouteDatum, SequenceDatum]):
        self._data = {route_id: constructor(value) for route_id, value in data.items()}

    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath) as f:
            data = json.load(f)
        assert data
        return cls(data)

    def get_route(self, route_id):
        return self._data[route_id]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, route_id):
        return self.get_route(route_id)


class SequenceData(AmazonData):
    def __init__(self, data):
        super().__init__(data, SequenceDatum)


class RouteData(AmazonData):
    def __init__(self, data):
        super().__init__(data, RouteDatum)

    def get_routes_with_score_ids(self, scores_list):
        res = [
            route_id
            for route_id, route in self._data.items()
            if route.get_score() in scores_list
        ]
        return res

    def make_station_code_indxes(self):
        station_codes_dict = dict()
        station_index = 0
        for route_id, _ in self._data.items():
            station_code = self._data[route_id]._data["station_code"]
            if station_code not in station_codes_dict:
                station_codes_dict[station_code] = station_index
                station_index += 1
        return station_codes_dict

    def get_max_num_zones(self):
        max_num_zones = 0
        for route_id, _ in self._data.items():
            zone_len = len(set(self._data[route_id].zone_d.values()))
            max_num_zones = max(zone_len, max_num_zones)
        return max_num_zones


class TravelTimeData(AmazonData):
    def __init__(self, data):
        super().__init__(data, TravelTimeDatum)


class PackageData(AmazonData):
    def __init__(self, data):
        super().__init__(data, PackageDatum)
