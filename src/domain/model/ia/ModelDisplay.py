from abc import abstractmethod
from dataclasses import dataclass

from src.domain.model.ia.BaseAIModel import BaseAIModel


@dataclass(frozen=True)
class AIModelParameters:
    pass


@dataclass(frozen=True)
class PerformanceMetrics:
    pass


@dataclass(frozen=True)
class SerializedAIModel:
    @abstractmethod
    def deserialize(self) -> BaseAIModel: ...


@dataclass(frozen=True)
class IAModelDescription:
    parameters: AIModelParameters
    performance_metrics: PerformanceMetrics
