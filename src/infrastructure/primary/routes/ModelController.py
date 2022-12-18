import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from domain.EventHandler import EventHandler
from domain.model.ModelCommands import RetrainModel, AddWineToTrainingData
from domain.model.ModelEvent import ModelRetrained, WineAdded
from domain.prediction.Wine import *
from src.domain.model.ModelCommandHandler import ModelCommandHandler
from src.infrastructure.secondary.repositories.Model import CSVModelRepository

router = APIRouter(
    prefix="/model",
    tags=["model"]
)


class ModelInput(BaseModel):
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
    quality: float


class ModelController:
    def __init__(self):
        self.model_repository = CSVModelRepository()
        self.model_command_handler = ModelCommandHandler(
            model_repository=self.model_repository,
            model=self.model_repository.load()
        )


    @router.get("/")
    async def get_model(self):
        return self.model_command_handler.model.serialize()

    @router.get("/description")
    async def get_description(self):
        return self.model_command_handler.model.describe()

    @router.put("/")
    async def add_model_entry(self, wine: ModelInput):
        event: WineAdded = self.model_command_handler.onModelCommand(AddWineToTrainingData(
            quality=wine.quality,
            wine=Wine(
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
        ))
        return event.wine


    @router.post("/retrain")
    async def retrain_model(self):
        event: ModelRetrained = self.model_command_handler.onModelCommand(RetrainModel())
        self.model_command_handler = ModelCommandHandler(
            model_repository=self.model_repository,
            model=event.model
        )
        return event.model.serialize()
