import os
import json
import numpy as np
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
    points = np.array(points, dtype=np.int64)
    points = np.reshape(points, (-1, 2))
    scans[file_name] = points


def compute_distances_no_loops(pt, cloud):
    dists = -2 * np.dot(pt, cloud.T) + np.sum(cloud**2,
                                              axis=1) + np.sum(pt**2, axis=1)[:, np.newaxis]
    dists = np.sqrt(dists)
    return dists


cloud = scans['200305']
point = scans['200320']

dis = compute_distances_no_loops(point, cloud)
result = np.where(dis == np.amin(dis,axis= 1))

listOfCordinates = result[0]
print (listOfCordinates)

print("\n")
