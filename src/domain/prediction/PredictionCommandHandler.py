from dataclasses import dataclass

from src.domain.model.ia.BaseAIModel import BestWinePredictorAIModel, QualityPredictorAIModel
from src.domain.prediction import PredictionEvent
from src.domain.prediction.PredictionCommands import PredictionCommand, PredictQualityForWine, PredictBestWine
from src.domain.prediction.PredictionEvent import BestWinePredicted, QualityForWinePredicted
from src.domain.prediction.Wine import Wine


@dataclass(frozen=True)
class PredictionCommandHandler:
    best_wine_predictor: BestWinePredictorAIModel
    quality_predictor: QualityPredictorAIModel

    def onPredictQualityForWine(self, wine: Wine) -> QualityForWinePredicted:
        return QualityForWinePredicted(
            quality=self.quality_predictor.predict(wine)
        )

    def onPredictBestWine(self) -> BestWinePredicted:
        return BestWinePredicted(
            best_wine=self.best_wine_predictor.predict()
        )

    def onPredictionCommand(self, command: PredictionCommand) -> PredictionEvent:
        match command:
            case PredictQualityForWine(wine):
                return self.onPredictQualityForWine(wine)
            case PredictBestWine():
                return self.onPredictBestWine()
