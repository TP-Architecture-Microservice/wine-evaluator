from dataclasses import dataclass
from typing import Union

from src.domain.prediction.Wine import Wine, WineQuality


@dataclass(frozen=True)
class AddWineToTrainingData:
    wine: Wine
    quality: WineQuality


RetrainModel = object

ModelCommand = Union[AddWineToTrainingData, RetrainModel]
