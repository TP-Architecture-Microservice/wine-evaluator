from fastapi import APIRouter

router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)


@router.get("/")
async def predict_best_wine():
    pass


@router.post("/")
async def predict_wine_quality():
    pass
