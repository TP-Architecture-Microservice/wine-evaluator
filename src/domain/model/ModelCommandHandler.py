from dataclasses import dataclass

from src.domain.model.ModelCommands import ModelCommand, RetrainModel, AddWineToTrainingData
from src.domain.model.ModelEvent import ModelEvent, ModelRetrained, WineAdded
from src.domain.model.ModelRepostory import ModelRepository
from src.domain.model.ia.BaseAIModel import BaseAIModel
from src.domain.prediction.Wine import Wine, WineQuality, WineWithQuality


@dataclass()
class ModelCommandHandler:
    model_repository: ModelRepository
    model: BaseAIModel

    def onAddWineToTrainingData(self, wine: Wine, quality: WineQuality) -> WineAdded:
        wine_with_quality = WineWithQuality(wine, quality)
        self.model_repository.add_entry(wine_with_quality)
        return WineAdded(wine_with_quality)

    def onRetrainModel(self) -> ModelRetrained:
        model = self.model.retrain()
        self.model_repository.persist(model)
        self.model = model
        return ModelRetrained

    def onModelCommand(self, command: ModelCommand) -> ModelEvent:
        match command:
            case AddWineToTrainingData(wine, quality):
                return self.onAddWineToTrainingData(wine, quality)
            case RetrainModel():
                return self.onRetrainModel()
