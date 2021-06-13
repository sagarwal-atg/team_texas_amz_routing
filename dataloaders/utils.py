from enum import Enum, auto

OKRED = "\033[91m"
OKBLUE = "\033[94m"
ENDC = "\033[0m"
OKGREEN = "\033[92m"
OKYELLOW = "\033[93m"


class TrainTest(Enum):
    train = auto()
    test = auto()


class RouteScoreType(Enum):
    Low = auto()
    Medium = auto()
    High = auto()
