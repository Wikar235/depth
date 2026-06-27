import streamlit as st
import requests
import numpy as np
import open3d as o3d
from app.utils import visualize_ply_file, display_inlier_outlier

API_URL = "http://localhost:9000/predict"

def arrays_to_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

st.title("Depth Anything 3 — Point Cloud Demo")

uploaded = st.file_uploader("Upload images", type=["jpg"], accept_multiple_files=True)

if uploaded:
    if st.button("Run pipeline"):
        with st.spinner("Running model..."):
            response = requests.post(
                API_URL,
                files=[("files", (f.name, f.getvalue(), "image/jpeg")) for f in uploaded],
            )
            response.raise_for_status()
            data = response.json()

        total = data["total"]
        n_in  = data["n_in"]
        n_out = data["n_out"]
        pcd           = arrays_to_pcd(data["inlier_points"],   data["inlier_colors"])
        inlier_cloud  = arrays_to_pcd(data["inlier_points"],   data["inlier_colors"])
        outlier_cloud = arrays_to_pcd(data["outlier_points"],  data["outlier_colors"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total points", f"{total:,}")
        col2.metric("Inliers", f"{n_in:,}", f"{100*n_in/total:.1f}%")
        col3.metric("Outliers", f"{n_out:,}", f"-{100*n_out/total:.1f}%")

        display_inlier_outlier(inlier_cloud, outlier_cloud)
        visualize_ply_file(pcd)