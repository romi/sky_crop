import os
import json
import numpy as np
import open3d as o3d
import copy
from _parameters_ import folder

print("\n")

base_folder = os.path.split(folder)[0]

json_dirs = []

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith("_coordinates.json"):
            json_dirs.append((os.path.join(root, file)))
# print (json_dirs)

scans = {}  # {date : np.array(points)}
for file in json_dirs:
    file_name = os.path.split(file)[1].split('_')[0]
    points = []
    with open(file) as file:
        index_json = json.load(file)
        for key in index_json.keys():
            pt = index_json[key]
            points.append(pt['x'])
            points.append(pt['y'])
    points = np.array(points, dtype=np.int32)
    points = np.reshape(points, (-1, 2))
    scans[file_name] = points


scan1 = scans['200305']
scan2 = scans['200310']
scan3 = scans['200315']
scan4 = scans['200320']

# convert to a cloud to use open3d


def to_o3d_cloud(cloud):
    cloud_coord = np.insert(cloud, 2, 0, axis=1)
    cloud_3d = o3d.geometry.PointCloud()
    cloud_3d.points = o3d.utility.Vector3dVector(cloud_coord)
    return cloud_3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.5, 0])
    target_temp.paint_uniform_color([0, 0.5, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    draw_registration_result(source, target, np.identity(4))
    source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


def global_registration (source,target,voxel_size):
    global result_ransac
    source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    result_ransac = execute_global_registration(source, target,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    result_icp = refine_registration(
        source, target, source_fpfh, target_fpfh, voxel_size)
    draw_registration_result(source, target, result_icp.transformation)
    
    return result_icp

source = to_o3d_cloud(scan1)
target = to_o3d_cloud(scan4)
voxel_size = 5


global_registration(source,target,voxel_size)