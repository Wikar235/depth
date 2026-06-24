from app.model import load_model, run_model_inference
from app.point_cloud import merge_point_clouds, merge_points_colors_to_cloud, statistical_outlier_remover
from app.utils import save_uploaded_images

model_name="depth-anything/da3-base"
model = load_model(model_name)

def run_pipeline(uploaded_files, model, confidence_thresh=0.5, file_name="merged_pcd"):
    

    image_paths = save_uploaded_images(uploaded_files)

    prediction = run_model_inference(model, image_paths)

    merged_points, merged_colors = merge_point_clouds(prediction, confidence_thresh=confidence_thresh)

    pcd = merge_points_colors_to_cloud(merged_colors, merged_points)

    pcd, total, n_in, n_out, inlier_cloud, outlier_cloud = statistical_outlier_remover(pcd)

    return pcd, total, n_in, n_out, inlier_cloud, outlier_cloud