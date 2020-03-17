import cv2
import imutils
import json
import numpy as np
from _parameters_ import div_y, div_x

# take detected lettuces' position from "plant_index.json" file.
# assemble the points from cropped images to the large image.
# return global position of points on the large image.


def assemble_points(img_names):
    points = {}
    ind = 0
    for name in img_names:
        # print('\n')
        # print(name)
        ind_x = int(name[-5])
        ind_y = int(name[-7])
        sub_pts = img_names[name]
        for pt_ind in sub_pts:
            pt = sub_pts[pt_ind]
            # print('\t', float(pt['X']), float(pt['Y']))
            # print('\t', float(pt['range_x']) * ind_x, float(pt['range_y']) * ind_y)
            position_x = float(pt['X']) + float(pt['range_x']) * ind_x
            position_y = float(pt['Y']) + float(pt['range_y']) * ind_y
            # print('pt{}'.format(ind), '\t', position_x, position_y)
            point = {}
            point['x'] = position_x
            point['y'] = position_y
            points[ind] = point
            ind += 1
    return points


# path to stitched, aligned and detected image
img_dir = '/Users/soroush/Desktop/detected.jpg'
img = cv2.imread(img_dir, 3)

print("--------------------------------------------")
print("Started Indexing", '\n')

# path to plant_index.json
json_file = '/Users/soroush/Desktop/plant_index.json'
with open(json_file) as ind_file:
    img_names = json.load(ind_file)

points = assemble_points(img_names)
radius = 80
for pt_ind in points:
    coordinates = points[pt_ind]
    center = (int(coordinates['x']), int(coordinates['y']))
    # print(center)
    # cv2.circle(img, center, radius, (255, 255, 255), 3)
    text_center = (int(coordinates['x']-radius/2), int(coordinates['y']+radius/3))
    cv2.putText(img, str(pt_ind), text_center, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 15)

cv2.imwrite('indexed.jpg', img)