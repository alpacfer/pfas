import open3d as o3d
import numpy as np
import os
import copy
import glob

# User-configurable parameters
SKIP_INTERVAL = 10
MAX_POINTS = 10000
MAX_POINTS_FINAL = 500000
VOXEL_SIZE = 0.05
THRESHOLD = 0.5

# Load RGB and Depth Images
rgb_folder = "./car_challenge/rgb"
depth_folder = "./car_challenge/depth"

rgb_images = sorted([os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder) if f.endswith(".jpg")])
depth_images = sorted([os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith(".png")])

if len(rgb_images) == 0 or len(depth_images) == 0:
    raise ValueError("No RGB or depth images found in the specified folders.")

rgb_images = rgb_images[::SKIP_INTERVAL]
depth_images = depth_images[::SKIP_INTERVAL]

print("Starting to load RGB and depth images...")
print(f"RGB Images: {len(rgb_images)}, Depth Images: {len(depth_images)}")

# Camera Intrinsics
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=525, fy=525, cx=319.5, cy=239.5
)

# Helper function to draw registrations
def draw_registrations(source, target, transformation=None, recolor=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if recolor:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# Generate Point Clouds
point_clouds = []
print("Starting to generate point clouds from RGB-D images...")
for idx, (rgb_path, depth_path) in enumerate(zip(rgb_images, depth_images)):
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, convert_rgb_to_intensity=False, depth_trunc=3.0, depth_scale=1000.0
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsics
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE * 2)
    point_clouds.append(pcd)

print("Finished generating point clouds.")
print(f"Point Cloud Sizes: {[len(np.asarray(pcd.points)) for pcd in point_clouds]}")

# Perform pairwise registration and merge point clouds
merged_pcd = point_clouds[0]
print("Starting pairwise registration and merging point clouds...")
for i in range(1, len(point_clouds)):
    source = merged_pcd
    target = point_clouds[i]

    # Estimate normals for ICP
    print(f"Estimating normals for source point cloud {i}...")
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30))
    print(f"Estimating normals for target point cloud {i}...")
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30))

    # Perform ICP registration
    point_to_plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    print(f"Starting ICP registration for point cloud {i} with multi-scale approach...")
    for scale in [VOXEL_SIZE * 5, VOXEL_SIZE * 2, VOXEL_SIZE]:
        source_down = source.voxel_down_sample(voxel_size=scale)
        target_down = target.voxel_down_sample(voxel_size=scale)
        icp_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, THRESHOLD, np.eye(4), point_to_plane,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
        )
        result_global = icp_result.transformation

    print(f"ICP result for point cloud {i}: Fitness={icp_result.fitness}, Inlier RMSE={icp_result.inlier_rmse}")
    print(f"Transformation Matrix for point cloud {i}: \n{icp_result.transformation}\n")

    # Apply transformation
    print(f"Applying transformation to point cloud {i}...")
    target.transform(icp_result.transformation)

    # Merge point clouds
    print(f"Merging point cloud {i} with the accumulated point cloud...")
    merged_pcd += target
    attempts = 0
    max_attempts = 5
    voxel_size = VOXEL_SIZE
    while attempts < max_attempts:
        try:
            print(f"Downsampling merged point cloud with voxel size {voxel_size} (Attempt {attempts + 1})...")
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
            break
        except RuntimeError as e:
            print(f"Warning: Voxel size too small for downsampling (Attempt {attempts + 1}). Increasing voxel size.")
            voxel_size *= 2
            attempts += 1
    if attempts == max_attempts:
        raise RuntimeError("Failed to downsample point cloud after multiple attempts. Consider increasing initial voxel size.")

# Save the final merged point cloud
final_output_file = "./car_challenge/final_merged.ply"
print("Saving final merged point cloud to file...")
o3d.io.write_point_cloud(final_output_file, merged_pcd)
print(f"Saved final merged point cloud to {final_output_file}")

# Visualize the final merged point cloud
o3d.visualization.draw_geometries([merged_pcd])