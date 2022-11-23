from dataclasses import dataclass
from typing import Union

from src.domain.prediction.Wine import Wine, WineWithQuality


@dataclass(frozen=True)
class WineAdded:
    wine: WineWithQuality


ModelRetrained = object

ModelEvent = Union[WineAdded, ModelRetrained]