import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from domain.model.ia.BaseAIModel import BestWinePredictorAIModel, QualityPredictorAIModel
from domain.prediction.PredictionCommandHandler import PredictionCommandHandler
from domain.prediction.PredictionCommands import PredictQualityForWine, PredictBestWine
from domain.prediction.PredictionEvent import QualityForWinePredicted
from domain.prediction.Wine import Wine

router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)


class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


class ModelController:
    def __init__(self):
        self.prediction_command_handler = PredictionCommandHandler(
            best_wine_predictor=BestWinePredictorAIModel(),
            quality_predictor=QualityPredictorAIModel()
        )

    @router.get("/")
    async def predict_best_wine(self):
        return self.prediction_command_handler.onPredictionCommand(PredictBestWine()).best_wine

    @router.post("/")
    async def predict_wine_quality(self, wine: WineInput):
        wine = Wine(
            id=uuid.uuid4(),
            fixed_acidity=wine.fixed_acidity,
            volatile_acidity=wine.volatile_acidity,
            citric_acid=wine.citric_acid,
            residual_sugar=wine.residual_sugar,
            chlorides=wine.chlorides,
            free_sulfur_dioxide=wine.free_sulfur_dioxide,
            total_sulfur_dioxide=wine.total_sulfur_dioxide,
            density=wine.density,
            pH=wine.pH,
            sulphates=wine.sulphates,
            alcohol=wine.alcohol,
        )
        event = self \
            .prediction_command_handler \
            .onPredictionCommand(PredictQualityForWine(wine))
        return event.quality
