import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import Quaternion, points_in_box
import open3d as o3d
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt
from PIL import Image


class Lidar_Processing:
    def __init__(self):
        self.nusc = NuScenes(version="v1.0-mini", dataroot="v1.0-mini", verbose=True)
        self.scene = self.nusc.scene[1]
        self.first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        self.current_sample = self.first_sample
        self.aggregated_pcd = o3d.geometry.PointCloud()
        self.color_value = np.zeros((0, 3))
        self.moving_points = o3d.geometry.PointCloud()
        self.static_points = o3d.geometry.PointCloud()
        self.initial_pose = np.array([0, 0, 0])
        self.trajectory_points = []
        self.camera_sensors = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
        ]
        self.all_lidar_pointclouds = []
        self.annotations = np.array([])

    def get_velocity(self, ann):
        """Get velocity of the object"""
        if "velocity" in ann:
            return np.linalg.norm(ann["velocity"], ord=2)
        elif ann["prev"]:
            # Calculate velocity using previous annotation
            current_translation = np.array(ann["translation"])
            current_timestamp = self.nusc.get("sample", ann["sample_token"])[
                "timestamp"
            ]
            prev_ann = self.nusc.get("sample_annotation", ann["prev"])
            prev_translation = np.array(prev_ann["translation"])
            prev_timestamp = self.nusc.get("sample", prev_ann["sample_token"])[
                "timestamp"
            ]
            displacement = current_translation - prev_translation
            time_delta = (
                current_timestamp - prev_timestamp
            ) / 1e6  # Convert microseconds to seconds
            return np.linalg.norm(displacement) / time_delta
        else:
            # No velocity info or no previous annotation
            return 0.0

    def detect_moving_objects(self, v_t=0.25):
        """Detect moving objects in the aggregated point cloud"""
        agg_pc = np.array(self.aggregated_pcd.points)
        agg_color = np.array(self.aggregated_pcd.colors)
        # Masks for moving and static points
        moving_mask = np.zeros(agg_pc.shape[0], dtype=bool)
        static_mask = np.ones(agg_pc.shape[0], dtype=bool)
        for ann in self.annotations:
            # Get velocity
            velocity = self.get_velocity(ann)
            is_moving = velocity >= v_t
            # Bounding box for object
            box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))
            # Get inside box points
            points_inside_box = points_in_box(box, agg_pc.T)
            if is_moving and points_inside_box.sum() >= 30:
                moving_mask[points_inside_box] = True
            else:
                static_mask[points_inside_box] = False

        moving_points = agg_pc[moving_mask]
        moving_colorvalue = agg_color[moving_mask]
        static_points = agg_pc[static_mask]
        static_colorvalue = agg_color[static_mask]
        self.moving_points.points = o3d.utility.Vector3dVector(moving_points)
        self.moving_points.colors = o3d.utility.Vector3dVector(moving_colorvalue)
        self.static_points.points = o3d.utility.Vector3dVector(static_points)
        self.static_points.colors = o3d.utility.Vector3dVector(static_colorvalue)

        return self.static_points, self.moving_points

    def transformation_matrix(self, transformation_data):
        """Create a transformation matrix from a transformation dictionary."""
        T = np.eye(4)
        q = Quaternion(transformation_data["rotation"])
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

    def camera_setup(self, camera_data, base_to_global):
        """Get camera extrinsic and intrinsic parameters"""
        camera_calibration = self.nusc.get(
            "calibrated_sensor", camera_data["calibrated_sensor_token"]
        )
        camera_intrinsic = np.array(camera_calibration["camera_intrinsic"])
        camera_to_base = self.transformation_matrix(camera_calibration)
        camera_to_global = np.dot(base_to_global, camera_to_base)
        camera_extrinsic = np.linalg.inv(camera_to_global)

        return camera_extrinsic, camera_intrinsic

    def project_point_to_image(self, points, camera_extrinsic, camera_intrinsic):
        """Project 3D points to 2D image plane"""
        points = np.vstack([points[:3, :], np.ones(points.shape[1])])
        camera_coordinate = np.dot(camera_extrinsic, points)
        depth = camera_coordinate[2, :]
        normalized_points = camera_coordinate[:3] / depth
        pixel_coordinate = np.matmul(camera_intrinsic, normalized_points)
        pixel_coordinate = pixel_coordinate[:2:]
        return pixel_coordinate, depth

    def draw_point_cloud_toimage(self, point_cloud_2d):

        self.camera_image = np.array(self.camera_image)
        for point in point_cloud_2d.T:
            x, y = int(point[0]), int(point[1])
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            self.camera_image[int(y), int(x)] = [255, 255, 255]

        img = Image.fromarray(self.camera_image)

    def colored_pointcloud(self, pointcloud, base_to_global, outfile=None):
        """Color the point cloud based on the camera images"""
        MAX_DEPTH = 9000
        DEPTH_THRESHOLD = 0.1
        pointcloud_data = np.array(pointcloud.points)
        point_rgb = np.full((pointcloud_data.shape[1], 3), 255, dtype=np.uint8)
        point_depth = np.full(pointcloud_data.shape[1], MAX_DEPTH, dtype=np.float32)
        min_depth = MAX_DEPTH
        max_depth = 0
        for camera in self.camera_sensors:
            camera_data = self.nusc.get(
                "sample_data", self.current_sample["data"][camera]
            )
            image_file = self.nusc.get_sample_data_path(camera_data["token"])
            image = np.array(Image.open(image_file))
            image_size = image.shape[:2]
            camera_extrinsic, camera_intrinsic = self.camera_setup(
                camera_data, base_to_global
            )
            # Save which point cloud point is mapped to a pixel
            pixel_point_map = np.full(image_size, 0, dtype=np.uint32)
            # If a point cloud point has been mapped to a pixel
            pixel_point_map_set = np.full(image_size, False, dtype=bool)
            pixel, depth = self.project_point_to_image(
                pointcloud_data, camera_extrinsic, camera_intrinsic
            )

            ind_in_image = np.where(
                (pixel[0] >= 0)
                & (pixel[0] < image_size[1])
                & (pixel[1] >= 0)
                & (pixel[1] < image_size[0])
                & (depth >= 0)
            )[0]
            for i, ind in enumerate(ind_in_image):
                pt = pixel[:, ind]
                x = int(pt[0])
                y = int(pt[1])
                min_depth = min(min_depth, depth[ind])
                max_depth = max(max_depth, depth[ind])
                if point_depth[ind] > depth[ind]:
                    point_depth[ind] = depth[ind]
                    # Map the point to a pixel if
                    # 1. The pixel has not been mapped to a point cloud point yet,
                    # 2. If the point is closer to the camera than the previous point that was mapped to this pixel
                    # 3. If the new point is within a certain depth range of the previous point that was mapped to this pixel
                    # Removing this may give you more colored points, at the expense of accuracy
                    if (
                        pixel_point_map_set[y][x] is False
                        or (point_depth[pixel_point_map[y][x]] > depth[ind])
                        or abs(point_depth[pixel_point_map[y][x]] - depth[ind]) < 0.5
                    ):
                        old_ind = pixel_point_map[y][x]
                        pixel_point_map[y][x] = ind
                        pixel_point_map_set[y][x] = True
                        point_rgb[ind] = image[y][x]
                        # If a point was already mapped to this pixel, but the new point is closer, then set the old point to white
                        if (point_depth[old_ind] > depth[ind]) and abs(
                            point_depth[old_ind] - depth[ind]
                        ) > 0.5:
                            point_rgb[old_ind] = [255, 255, 255]
        color = point_rgb / 255
        # Add the new columns to the existing array
        return color

    def lidar_data_setup(self):
        """Load point cloud data from the lidar sensor"""
        lidar_token = self.current_sample["data"]["LIDAR_TOP"]
        lidar_data = self.nusc.get("sample_data", lidar_token)
        # Load lidar data
        lidar_filepath = self.nusc.get_sample_data_path(lidar_token)
        self.pointcloud = LidarPointCloud.from_file(lidar_filepath)
        self.all_lidar_pointclouds.append(self.pointcloud)
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = self.nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )

        return self.pointcloud, ego_pose, calibrated_sensor

    def get_annotations(self):
        """Get annotations for the current sample"""
        for ann_token in self.current_sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            self.annotations = np.append(self.annotations, ann)
        # print(self.annotations)
        return self.annotations

    def cloud_aggregation(self):
        """Aggregate point clouds from multiple samples"""
        index = 0
        while self.current_sample:
            # Load lidar point cloud
            self.pointcloud, ego_pose, calibrated_sensor = self.lidar_data_setup()
            # Get annotations
            self.annotations = self.get_annotations()
            # transformation from lidar to base
            lidar_to_base = self.transformation_matrix(calibrated_sensor)
            # Transform to global frame
            base_to_global = self.transformation_matrix(ego_pose)
            self.pointcloud.transform(lidar_to_base)
            self.pointcloud.transform(base_to_global)
            color_value = self.colored_pointcloud(self.pointcloud, base_to_global)
            self.color_value = np.vstack([self.color_value, color_value])

            # print(self.color_value)
            # Add points to aggregated point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.pointcloud.points.T[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(color_value)

            self.aggregated_pcd += pcd
            self.trajectory_points.append(ego_pose["translation"])
            self.current_sample = (
                self.nusc.get("sample", self.current_sample["next"])
                if self.current_sample["next"]
                else None
            )
            index += 1

        return self.aggregated_pcd, self.color_value

    def plot_trajectory(self):
        trajectory_points = np.array(self.trajectory_points)
        trajectory_points = trajectory_points[:, :2]
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 1])
        plt.show()

    def visualize_filtred_pointcloud(
        self,
    ):
        self.moving_points.paint_uniform_color([1, 0, 0])
        self.static_points.paint_uniform_color([0, 1, 0])
        all_points = self.moving_points + self.static_points
        o3d.visualization.draw_geometries([all_points])

    def visualize_pointcloud(self):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.aggregated_pcd.points)
        o3d.io.write_point_cloud("aggregated_map.ply", pcd)
        o3d.visualization.draw_geometries([pcd])

    def visulized_clored_pointcloud(self):
        o3d.io.write_point_cloud("aggregated_map_color.ply", self.aggregated_pcd)
        o3d.visualization.draw_geometries([self.aggregated_pcd])

    def visualize_moving(self):
        self.moving_points.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([self.moving_points])

    def visualize_static(self):
        self.static_points.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([self.static_points])

    def visualize(self, pcd):

        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    process = Lidar_Processing()
    aggregated_cloud, color = process.cloud_aggregation()
    process.visualize_pointcloud()
    static_points, moving_points = process.detect_moving_objects()
    process.visualize(static_points)
    # process.visualize_filtred_pointcloud()
    # process.visualize_moving()
    # process.visualize_static()
