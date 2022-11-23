from abc import abstractmethod

from src.domain.model.ia.ModelDisplay import IAModelDescription, SerializedAIModel
from src.domain.prediction.Wine import WineQuality, Wine


class BaseAIModel:
    @abstractmethod
    def describe(self) -> IAModelDescription: ...

    @abstractmethod
    def serialize(self) -> SerializedAIModel: ...

    @abstractmethod
    def retrain(self) -> "BaseAIModel": ...


class QualityPredictorAIModel(BaseAIModel):
    @abstractmethod
    def predict(self, wine: Wine) -> WineQuality: ...


class BestWinePredictorAIModel(BaseAIModel):
    @abstractmethod
    def predict(self) -> Wine: ...
