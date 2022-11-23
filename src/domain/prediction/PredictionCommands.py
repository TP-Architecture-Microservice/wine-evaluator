from dataclasses import dataclass
from typing import Union

from src.domain.prediction.Wine import Wine


@dataclass(frozen=True)
class PredictQualityForWine:
    wine: Wine


PredictBestWine = object

PredictionCommand = Union[PredictBestWine, PredictQualityForWine]
