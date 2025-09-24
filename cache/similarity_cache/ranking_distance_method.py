from enum import Enum, auto


class RankingDistanceMethod(Enum):
    EUCLIDEAN = auto()
    MANHATTAN = auto()
    COSINE = auto()
