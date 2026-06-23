import numpy as np
import open3d as o3d
import streamlit as st

def create_point_cloud(depth_map, rgb_image, intrinsics, extrinsics, confidence_map, confidence_thresh=0.5):
    h, w = depth_map.shape

    # Standard pinhole projection math (Creates 2D grids)
    # intrinsics describes the camera itself (lens/sensor properties)
    # fx, fy are the focal lengths (in pixels)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]

    # cx, cy are the principal point - basically the pixel coordinates of the image center
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    # Grid shape: (H, W, 3)
    points_cam = np.stack([x, y, z], axis=-1)
    
    # Color grid shape: (H, W, 3) normalized between 0.0 and 1.0
    colors_grid = rgb_image.astype(np.float32) / 255.0

    # 2. THE THRESHOLD LOGIC
    # Create a 2D boolean mask where confidence is high AND depth is valid (not 0)
    valid_mask = (confidence_map > confidence_thresh) & (depth_map > 0)

    # 3. Flatten the grids into simple lists using the mask
    # This automatically converts the data from (H, W, 3) to (N, 3)
    points_cam_flat = points_cam[valid_mask]  # Shape: (N, 3)
    colors_flat = colors_grid[valid_mask]      # Shape: (N, 3)

    # 4. Transform from Camera Space to World Space
    # Extrinsics broken into Rotation (3x3) and Translation (3,)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    
    # Math correction: Shifting points from camera origin to world origin
    # For a flat list of points (N, 3), the matrix math looks like this:
    points_world = (points_cam_flat - t) @ R

    return points_world, colors_flat

def merge_point_clouds(prediction, confidence_thresh=0.5):
    """merge point cloud into one object"""

    print('before')
    all_points=[]
    all_colors=[]

    # number of images in depth map
    frames_count = len(prediction.depth)

    for i in range(frames_count):
        points, colors = create_point_cloud(
            prediction.depth[i],
            prediction.processed_images[i],
            prediction.intrinsics[i],
            prediction.extrinsics[i],
            prediction.conf[i],
            confidence_thresh
        )
        all_points.append(points)
        all_colors.append(colors)

    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    print(f"Total points in cloud: {len(merged_points)}")
    
    return merged_points, merged_colors

def merge_points_colors_to_cloud(merged_colors, merged_points):
    # Instantiate an empty Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Assign your numpy coordinates (Vector3dVector converts them to Open3D format)
    pcd.points = o3d.utility.Vector3dVector((merged_points).astype(np.float32))

    # Assign your normalized colors
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    print(f"Successfully saved {len(merged_points)} points")

    return pcd

def statistical_outlier_remover(pcd):

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)

    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    total = len(pcd.points)
    n_in  = len(inlier_cloud.points)
    n_out = len(outlier_cloud.points)

    print(f"Total:    {total:,}")
    print(f"Inliers:  {n_in:,}  ({100*n_in/total:.1f}%)")
    print(f"Outliers: {n_out:,} ({100*n_out/total:.1f}%)")

    return pcd, total, n_in, n_out, inlier_cloud, outlier_cloud

