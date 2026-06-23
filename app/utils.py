import tempfile
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def save_uploaded_images(uploaded_files):
    tmp_dir = tempfile.mkdtemp()
    image_paths = []

    for f in uploaded_files:
        path = os.path.join(tmp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        image_paths.append(path)

    print(f"Saved {len(image_paths)} images to {tmp_dir}")
    return image_paths

def visualize_ply_file(pcd):

    pts = np.asarray(pcd.points)
    cols = (np.asarray(pcd.colors) * 255).astype(int)

    # Sample for performance
    idx = np.random.choice(len(pts), 50000, replace=False)

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[idx, 0],
        y=pts[idx, 1],
        z=pts[idx, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=[f'rgb({r},{g},{b})' for r, g, b in cols[idx]],
        )
    )])

    fig.update_layout(
    scene=dict(aspectmode='data'),
    height=1000
    )
    st.plotly_chart(fig, use_container_width=True)

def display_inlier_outlier(inlier_cloud, outlier_cloud):

    in_pts  = np.asarray(inlier_cloud.points)
    out_pts = np.asarray(outlier_cloud.points)

    # Sample if large
    def sample(pts, n=50000):
        idx = np.random.choice(len(pts), min(n, len(pts)), replace=False)
        return pts[idx]

    in_pts  = sample(in_pts)
    out_pts = sample(out_pts)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=in_pts[:, 0], y=in_pts[:, 1], z=in_pts[:, 2],
            mode='markers', name=f'Inliers ({len(in_pts)})',
            marker=dict(size=2, color='blue')
        ),
        go.Scatter3d(
            x=out_pts[:, 0], y=out_pts[:, 1], z=out_pts[:, 2],
            mode='markers', name=f'Outliers ({len(out_pts)})',
            marker=dict(size=0.5, color='red')
        )
    ])

    fig.update_layout(
    scene=dict(aspectmode='data'),
    height=1000
    )
    st.plotly_chart(fig, use_container_width=True)