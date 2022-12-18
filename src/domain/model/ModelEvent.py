from dataclasses import dataclass
from typing import Union

from domain.model.ia.BaseAIModel import BaseAIModel
from src.domain.prediction.Wine import WineWithQuality


@dataclass(frozen=True)
class WineAdded:
    wine: WineWithQuality

@dataclass(frozen=True)
class ModelRetrained:
    model: BaseAIModel


ModelEvent = Union[WineAdded, ModelRetrained]
