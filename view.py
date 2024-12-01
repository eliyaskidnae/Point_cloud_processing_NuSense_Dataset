from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np
import open3d as o3d

# Load dataset
nusc = NuScenes(version="v1.0-mini", dataroot="v1.0-mini", verbose=True)

# # Choose a scene
scene = nusc.scene[0]  # Select a specific scene
print(scene)
first_sample = nusc.get("sample", scene["first_sample_token"])
print(first_sample)

# # Initialize aggregated point cloud
# aggregated_pcd = o3d.geometry.PointCloud()

# Iterate through samples
current_sample = first_sample
while current_sample:
    lidar_token = current_sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    calibrated_sensor = nusc.get(
        "calibrated_sensor", lidar_data["calibrated_sensor_token"]
    )

    # Load lidar data
    lidar_filepath = nusc.get_sample_data_path(lidar_token)
    pointcloud = LidarPointCloud.from_file(lidar_filepath)

    # Transform to ego frame
    lidar_to_ego = transform_matrix(
        calibrated_sensor["translation"], calibrated_sensor["rotation"]
    )
    pointcloud.transform(lidar_to_ego)

    # Transform to global frame
    ego_to_global = transform_matrix(ego_pose["translation"], ego_pose["rotation"])
    pointcloud.transform(ego_to_global)

    # Add points to aggregated point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.points.T[:, :3])
    aggregated_pcd += pcd

    # Move to next sample
    current_sample = (
        nusc.get("sample", current_sample["next"]) if current_sample["next"] else None
    )

# # Save and visualize
# o3d.io.write_point_cloud("aggregated_map.ply", aggregated_pcd)
# o3d.visualization.draw_geometries([aggregated_pcd])
