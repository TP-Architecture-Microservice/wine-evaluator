from fastapi import FastAPI
from routes.ModelController import router as model_router
from routes.PredictController import router as predict_router


def run_api():
    app = FastAPI()
    app.include_router(model_router)
    app.include_router(predict_router)
