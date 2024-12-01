import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, Quaternion
import open3d as o3d
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt


class Lidar_Processing:
    def __init__(self):
        self.nusc = NuScenes(version="v1.0-mini", dataroot="v1.0-mini", verbose=True)
        self.scene = self.nusc.scene[7]
        self.first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        self.current_sample = self.first_sample
        self.aggregated_pcd = o3d.geometry.PointCloud()
        self.initial_pose = np.array([0, 0, 0])
        self.trajectory_points = []
        self.moving_objects = o3d.geometry.PointCloud()

    def transformation_matrix(self, transformation_data):
        T = np.eye(4)
        q = Quaternion(transformation_data["rotation"])
        lidar_to_base = np.eye(4)
        T[:3, :3] = q.rotation_matrix
        T[:3, 3] = transformation_data["translation"]

        return T

    def simulate_moving_objects(self, ego_pose):
        # Simulate moving objects by adding points around the ego vehicle
        num_objects = 5
        radius = 10
        angles = np.linspace(0, 2 * np.pi, num_objects, endpoint=False)
        object_positions = np.array(
            [
                ego_pose["translation"]
                + radius * np.array([np.cos(angle), np.sin(angle), 1000])
                for angle in angles
            ]
        )
        return object_positions

    def cloud_aggregation(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        while self.current_sample:
            lidar_token = self.current_sample["data"]["LIDAR_TOP"]
            lidar_data = self.nusc.get("sample_data", lidar_token)
            ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
            calibrated_sensor = self.nusc.get(
                "calibrated_sensor", lidar_data["calibrated_sensor_token"]
            )

            # Load lidar data
            lidar_filepath = self.nusc.get_sample_data_path(lidar_token)
            self.pointcloud = LidarPointCloud.from_file(lidar_filepath)
            # Transform to base frame
            lidar_to_base = self.transformation_matrix(calibrated_sensor)
            self.pointcloud.transform(lidar_to_base)
            # Transform to global frame
            base_to_global = self.transformation_matrix(ego_pose)

            self.pointcloud.transform(base_to_global)
            # Add points to aggregated point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.pointcloud.points.T[:, :3])
            self.aggregated_pcd += pcd
            self.trajectory_points.append(ego_pose["translation"])

            # Simulate moving objects
            moving_object_positions = self.simulate_moving_objects(ego_pose)
            self.moving_objects.points = o3d.utility.Vector3dVector(
                moving_object_positions
            )

            # update window
            # Create a line set for the trajectory
            if len(self.trajectory_points) > 1:
                lines = [[i, i + 1] for i in range(len(self.trajectory_points) - 1)]
                trajectory_line_set = o3d.geometry.LineSet()
                trajectory_line_set.points = o3d.utility.Vector3dVector(
                    self.trajectory_points
                )
                trajectory_line_set.lines = o3d.utility.Vector2iVector(lines)

                # Update the visualizer
                vis.clear_geometries()
                vis.add_geometry(self.aggregated_pcd)
                vis.add_geometry(self.moving_objects)
                # vis.add_geometry(trajectory_line_set)
                vis.poll_events()
                vis.update_renderer()
            time.sleep(0.1)
            # Move to next sample
            self.current_sample = (
                self.nusc.get("sample", self.current_sample["next"])
                if self.current_sample["next"]
                else None
            )

        vis.destroy_window()

        o3d.io.write_point_cloud("aggregated_map.ply", self.aggregated_pcd)

        return self.aggregated_pcd

    def plot_trajectory(self):
        trajectory_points = np.array(self.trajectory_points)
        trajectory_points = trajectory_points[:, :2]
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 1])
        plt.show()


if __name__ == "__main__":
    process = Lidar_Processing()
    aggregated_cloud = process.cloud_aggregation()
    o3d.visualization.draw_geometries([aggregated_cloud])
    process.plot_trajectory()
