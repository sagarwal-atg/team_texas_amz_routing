
from types import SimpleNamespace
import json
from typing import Dict, List, Sequence, Union
import numpy as np
from munch import Munch # used to give dot accessing to dict

class RouteDatum:
    SCORE = SimpleNamespace(LOW='Low', MEDIUM='Medium', HIGH='High')

    def __init__(self, data):
        self._data = Munch(data)

    def get_stops(self, stop_ids: List[str]=None):
        stops = self._data.stops
        if stop_ids:
            stops = {key: stops[key] for key in stop_ids}
        return stops

    def get_zones(self, stop_ids: List[str]=None) -> Dict[str, str]:
        """
        Returns: a map of zone id's for each stop
            ex: {'A': 'zone1', 'B': 'zone2', 'C': 'zone3'}
        """
        return {key: stop['zone_id'] for key, stop in self.get_stops(stop_ids).items()}

    def get_score(self):
        return self._data.route_score

    def is_high_score(self):
        return self.get_score() == self.SCORE.HIGH

class SequenceDatum:
    """
    Data about the actual route taken by the driver
    """
    def __init__(self, data: Dict[str, Dict[str, int]]):
        """
        data: a dict whose values indicate the order we visited the stops
            ex: {'actual': {'A': 0, 'B': 2, 'C': 1}}
        """
        self._stops = data['actual']

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

    def get_route_order(self):
        """
        stop_sequence: a dict whose values indicate the order we visited the stops
            ex: {'A': 0, 'B': 2, 'C': 1}
        Returns: a list of ints whose index is the FROM index in stop_sequence and whose value is
            the TO index in stop_sequence.
            ex: [2, 0, 1] (meaning that we go from [A->C, B->A, C->B])
        """
        stops_ordered = self._get_actual_order()
        next_stop = {stop: next_stop for stop, next_stop in zip(np.roll(stops_ordered, 1), stops_ordered)}
        stop_idx = {k: i for i, k in enumerate(self._stops)}
        labels = [stop_idx[next_stop[stop]] for stop in stop_idx.keys()] 
        return labels

    def __len__(self):
        return len(self.get_stop_ids())


class TravelTimeDatum:
    def __init__(self, data: Dict[str, Dict[str, float]]) -> None:
        self._data = data

    def as_matrix(self, stop_ids):
        route_len = len(stop_ids)
        times = np.zeros((route_len, route_len))
        for i, stop1 in enumerate(stop_ids):
            for j, stop2 in enumerate(stop_ids):
                times[i][j] = self._data[stop1][stop2]
        return times


class AmazonData:
    def __init__(self, data, constructor: Union[RouteDatum, SequenceDatum]):
        self._data = {route_id: constructor(value) for route_id, value in data.items()}

    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath) as f:
            data = json.load(f)
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

    def get_high_score_ids(self):
        return [route_id for route_id, route in self._data.items() if route.is_high_score()]

class TravelTimeData(AmazonData):
    def __init__(self, data):
        super().__init__(data, TravelTimeDatum)
