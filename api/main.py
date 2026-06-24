from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.model import load_model, run_model_inference

model_name="depth-anything/da3-base"
ml_models = {}

# https://fastapi.tiangolo.com/advanced/events/
# docs for asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["da-base"] = load_model(model_name)
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/predict")
async def predict(x: float):
    result = ml_models["da-base"](x)
    return {"result": result}
