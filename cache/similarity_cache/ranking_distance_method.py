from enum import StrEnum


class RankingDistanceMethod(StrEnum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
