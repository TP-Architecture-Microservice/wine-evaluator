from abc import abstractmethod

from src.domain.model.ia.BaseAIModel import BaseAIModel
from src.domain.prediction.Wine import WineWithQuality


class ModelRepository:
    @abstractmethod
    def persist(self, model: BaseAIModel): ...

    @abstractmethod
    def load(self) -> BaseAIModel: ...

    @abstractmethod
    def add_entry(self, wine_with_quality: WineWithQuality): ...
