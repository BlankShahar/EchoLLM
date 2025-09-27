from enum import Enum, auto


class RankingDistanceMethod(Enum):
    EUCLIDEAN = auto()
    MANHATTAN = auto()
    COSINE = auto() # Note: Max distance is 1
