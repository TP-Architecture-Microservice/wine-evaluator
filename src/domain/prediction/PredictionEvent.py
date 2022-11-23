from dataclasses import dataclass
from typing import Union

from src.domain.prediction.Wine import WineQuality, Wine


@dataclass(frozen=True)
class QualityForWinePredicted:
    quality: WineQuality


@dataclass(frozen=True)
class BestWinePredicted:
    best_wine: Wine


PredictionEvent = Union[QualityForWinePredicted, BestWinePredicted]