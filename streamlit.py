import streamlit as st
from app.pipe import run_pipeline
from app.utils import visualize_ply_file, display_inlier_outlier

st.title("Depth Anything 3 — Point Cloud Demo")

uploaded = st.file_uploader("Upload images", type=["jpg"], accept_multiple_files=True)

if uploaded:
    if st.button("Run pipeline"):
        with st.spinner("Running model..."):
            pcd, total, n_in, n_out, inlier_cloud, outlier_cloud = run_pipeline(uploaded)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total points", f"{total:,}")
        col2.metric("Inliers", f"{n_in:,}", f"{100*n_in/total:.1f}%")
        col3.metric("Outliers", f"{n_out:,}", f"-{100*n_out/total:.1f}%")

        visualize_ply_file(pcd)
        display_inlier_outlier(inlier_cloud, outlier_cloud)