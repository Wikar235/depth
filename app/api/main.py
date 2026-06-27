from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
import tempfile
import os
import numpy as np 

from app.model import load_model, run_model_inference
from app.point_cloud import merge_point_clouds, merge_points_colors_to_cloud, statistical_outlier_remover

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

# https://docs.python.org/3/library/tempfile.html
# tempfile documentation
# https://dev.to/nadun96/understanding-pythons-temporarydirectory-mec
# tempfile.TemporaryDirectory() usage
@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):

    file_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f'Temporary directory created at: {tmpdir}')

        for file in files:
            file_path = os.path.join(tmpdir, file.filename)
            print(file_path)

            content = await file.read()

            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(content)

            file_paths.append(file_path)

        model = ml_models["da-base"]
        prediction = run_model_inference(model, file_paths)

        merged_points, merged_colors = merge_point_clouds(prediction, confidence_thresh=0.5)

        pcd = merge_points_colors_to_cloud(merged_colors, merged_points)

        pcd, total, n_in, n_out, inlier_cloud, outlier_cloud = statistical_outlier_remover(pcd)

    return {
        "total": total,
        "n_in": n_in,
        "n_out": n_out,
        "inlier_points": np.asarray(inlier_cloud.points).tolist(),
        "inlier_colors": np.asarray(inlier_cloud.colors).tolist(),
        "outlier_points": np.asarray(outlier_cloud.points).tolist(),
        "outlier_colors": np.asarray(outlier_cloud.colors).tolist(),
    }
