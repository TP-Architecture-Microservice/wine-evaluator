from dataclasses import dataclass
from typing import Union

from src.domain.prediction.Wine import Wine, WineQuality


@dataclass(frozen=True)
class AddWineToTrainingData:
    wine: Wine
    quality: WineQuality


@dataclass(frozen=True)
class RetrainModel:
    pass


ModelCommand = Union[AddWineToTrainingData, RetrainModel]
