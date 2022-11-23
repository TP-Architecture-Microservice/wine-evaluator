from fastapi import APIRouter

from src.domain.model.ModelCommandHandler import ModelCommandHandler
from src.infrastructure.secondary.repositories.Model import CSVModelRepository

router = APIRouter(
    prefix="/model",
    tags=["model"]
)


model_repository = CSVModelRepository()

model_command_handler = ModelCommandHandler(
    model_repository=model_repository,
    model=model_repository.load()
)

@router.get("/")
async def get_model():
    pass

@router.get("/description")
async def get_description():
    pass

@router.put("/")
async def add_model_entry():
    pass

@router.post("/retrain")
async def retrain_model():
    pass