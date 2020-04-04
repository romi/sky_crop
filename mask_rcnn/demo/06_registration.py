import os
import json
import numpy as np
import open3d as o3d
import copy
import cv2
import imutils
from _parameters_ import folder


def to_o3d_cloud(cloud):
    # INPUT :: cloud: 2d numpy array [x,y]
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
    o3d.visualization.draw_geometries([source_temp, target_temp],width=1300, height=800)


def preprocess_point_cloud(pcd, voxel_size):
    radius_normal = voxel_size * 5
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 10
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    # print(":: Load two point clouds.")
    # draw_registration_result(source, target, np.identity(4))
    source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2
    # print(":: RANSAC registration.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)

    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


def global_registration(source, target, voxel_size):
    global result_ransac
    source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    result_ransac = execute_global_registration(source, target,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    result_icp = refine_registration(
        source, target, source_fpfh, target_fpfh, voxel_size)
    draw_registration_result(source, target, result_icp.transformation)

    return result_icp


def compute_distances(source, target):
    dists = -2 * np.dot(source, target.T) + np.sum(target**2,
                                                   axis=1) + np.sum(source**2, axis=1)[:, np.newaxis]
    dists = np.sqrt(dists)
    return dists


def closest_point(source_o3d, target_o3d):
    # Finding the closest point for each point in reference point-cloud to points from each scan:
    # 1. Finding the distances between points
    source_np = np.asarray(source_o3d.points)
    target_np = np.asarray(target_o3d.points)

    distances = compute_distances(source_np, target_np)

    # 2. Finding the index of closest pair of points
    min_distance = np.amin(distances, axis=1)  # per row
    min_distance_reshaped = np.reshape(min_distance, (-1, 1))  # grafting

    ind_source, ind_target = np.where(distances == min_distance_reshaped)
    ind_target_reshaped = np.reshape(ind_target, (-1, 1))  # grafting
    # print ('ind target\t', ind_target)

    return ind_target, min_distance


def find_markers(image):
    resized = imutils.resize(image, width=2000)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)  # converting to HSV
    lower_red = np.array([70, 80, 50])  # lower range for blue marker
    upper_red = np.array([150, 255, 255])  # upper range for blue marker
    mask = cv2.inRange(hsv, lower_red, upper_red)  # find the blue marker
    res = cv2.bitwise_and(resized, resized, mask=mask)
    blurred = cv2.GaussianBlur(res, (11, 11), cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    thresh = imutils.resize(thresh, width=image.shape[1])
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centroids = np.asarray([[0, 0, 0]])[1:]
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        markers = np.asarray([[cX, cY, 0]])
        centroids = np.concatenate((centroids, markers), axis=0)
        cv2.circle(thresh, (cX, cY), 30, (255, 255, 255), -1)
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    markers_cloud = o3d.utility.Vector3dVector(centroids)
    return markers_cloud, centroids


print("--------------------------------------------")
print("Started Registration", '\n')

# Reading scans from json files

base_folder = os.path.split(folder)[0]
json_dirs = []

detected_total_dir = {}
output_dir = {}

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith("_coordinates.json"):
            date = os.path.split(root)[-1]
            output_dir [date] = '{0}/{1}_registered.json'.format(root, date)
            detected_total_dir[date] = '{0}/{1}_detected_total.jpg'.format(
                (os.path.join(base_folder, date)), date)
            json_dirs.append((os.path.join(root, file)))

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


# Putting the best scan as reference and aligning everything on top of this


# Sorting scans by the number of scanned elements
dates = list(scans.keys())
len_scans = []
for i in dates:
    len_scans.append(len(scans[i]))
dates = [x for _, x in sorted(zip(len_scans, dates))]


marker_clouds = {}
# detecting the blue markers to use for registration
for i in detected_total_dir:
    img = cv2.imread(detected_total_dir[i])
    # img = imutils.resize(img, width=900)
    marker_cloud = o3d.geometry.PointCloud()
    marker_cloud.points, marker_centroids = find_markers(img)
    marker_clouds[i] = marker_cloud
    # print(marker_centroids)


voxel_size = 10

target = to_o3d_cloud(scans[dates[-1]])
markers_tg =  marker_clouds[dates[-1]]

global_reg = {}
pcd = {dates[-1]: target}


# Finding transformation matrices to register all pointclouds on top of target
for i in range(len(dates)-1):
    source = to_o3d_cloud(scans[dates[i]])
    pcd[dates[i]] = source
    markers_src = marker_clouds[dates[i]] 
    icp_res = global_registration(markers_src, markers_tg, voxel_size)
    global_reg[dates[i]] = icp_res.transformation
global_reg[dates[i]] = icp_res.transformation
global_reg[dates[len(dates)-1]] = np.identity(4)

cloud_total = o3d.geometry.PointCloud()

# Using the transformation Matrix to align the point clouds to reference point-cloud
registered = {}
for key in global_reg:
    global_reg[key] = np.asarray(global_reg[key], dtype=np.int16)
    registered[key] = pcd[key].transform(global_reg[key])
    cloud_total = cloud_total + registered[key]


# input: source is the reference point cloud, target is the could to compare

indexes = {}
target_o3d = registered[list(registered.keys())[-1]]
for date in registered:
    a = {}
    source_o3d = registered[date]
    index, distances = closest_point(source_o3d, target_o3d)
    # index[distances > 250] = -1
    # cp_result = np.column_stack((index,distances))
    index_list = index.tolist()
    for i,j in enumerate(index_list):
        a [i] = j
    indexes[date] = a

print (indexes)
o3d.visualization.draw_geometries([cloud_total], width=1300, height=800)


for i in output_dir:
    with open(output_dir[i], 'w') as outfile:
        json.dump(indexes[i], outfile)
