import os
import json
import numpy as np
import open3d
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


def compute_distances(pt, cloud):
    dists = -2 * np.dot(pt, cloud.T) + np.sum(cloud**2,
                                              axis=1) + np.sum(pt**2, axis=1)[:, np.newaxis]
    dists = np.sqrt(dists)
    return dists


scan1 = scans['200305']
scan2 = scans['200310']
scan3 = scans['200315']
scan4 = scans['200320']

# dis = compute_distances(point, cloud)
# result = np.where(dis == np.amin(dis,axis= 1))

# listOfCordinates = result[0]
# print (listOfCordinates)


# convert to a cloud to use open3d
def to_o3d_cloud(cloud):
    cloud_coord = np.insert(cloud, 2, 0, axis=1)
    cloud_3d = open3d.geometry.PointCloud()
    cloud_3d.points = open3d.utility.Vector3dVector(cloud_coord)
    return cloud_3d


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])


scan1_3d = to_o3d_cloud(scan1)
scan2_3d = to_o3d_cloud(scan3)


threshold = 300
# draw_registration_result(scan1_3d, scan2_3d)

print('Initial alignment\n')
print('source\t',len(scan1_3d.points),'points')
print('target\t',len(scan2_3d.points),'points')

evaluation = open3d.registration.evaluate_registration(scan1_3d, scan2_3d, threshold)

print("\nApply point-to-point ICP\n")
cor =evaluation.correspondence_set
print ('correspondence_set\t', np.asarray(cor))
print("\n")
print ('transformation\t', evaluation.transformation)
print("\n")
print ('fitness\t', evaluation.fitness)
print("\n")
print ('inlier_rmse\t', evaluation.inlier_rmse)
print("\n")



"""
reg_p2p = open3d.registration.registration_icp(scan1_3d, scan2_3d, threshold, trans_init,open3d.registration.TransformationEstimationPointToPoint())

print(reg_p2p)
print("\nTransformation is:")
print(reg_p2p.transformation)
print("")

draw_registration_result(scan1_3d, scan2_3d, reg_p2p.transformation)

print("\n")
"""