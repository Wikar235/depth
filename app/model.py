import torch
from depth_anything_3.api import DepthAnything3

def load_model(model_name="depth-anything/da3-base"):
    # 1. Setup GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model from Hugging Face
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)

    print(f"Model loaded: {model_name} on {device}")
    return model, device

def run_model_inference(model, image_paths,conf_thresh_percentile=50, export_dir=None):
    # Run inference
    prediction = model.inference(
        image=image_paths,
        # infer_gs=True,
        export_dir=export_dir,
        # export_format="glb_gs_ply",
        conf_thresh_percentile=conf_thresh_percentile,
    )

    print("running model inference")
    print(f"saved glb file to {export_dir}")

    return prediction