import numpy as np


def assemble_points(points, ind_x, ind_y):
    assembled_points = []
    for pt in points:
        position_x = float(pt['X']) + float(pt['range_x']) * ind_x
        position_y = float(pt['Y']) + float(pt['range_y']) * ind_y
        point = [position_x, position_y]
        assembled_points.append(point)
    assembled_points = np.asarray(assembled_points, dtype=np.int32)
    return assembled_points

