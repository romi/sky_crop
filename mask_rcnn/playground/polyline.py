from imutils import paths
import numpy as np
import argparse
import imutils
import random
import json
import cv2
import os

path = "/Users/soroush/Desktop/Noumena/sky_crop/mask_rcnn/annotated/via_project_6Dec2019_15h1m.json"

# parse JSON
with open(path, 'r') as myfile:
    data = myfile.read()
json_data = json.loads(data)
metadata =  (json_data['_via_img_metadata'])
for i in metadata:
    shape_attributes = (metadata[i]['regions'][0]['shape_attributes'])
    x = shape_attributes['all_points_x']
    y = shape_attributes['all_points_y']


# read source image
img = cv2.imread('/Users/soroush/Desktop/Noumena/sky_crop/mask_rcnn/annotated/10.JPG')
b_channel, g_channel, r_channel = cv2.split(img)

# draw white mask on black background
background = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
ptsList  = np.column_stack((x,y))
cv2.polylines(background, [ptsList], True, (0, 0, 0), 1)
cv2.fillPoly(background, [ptsList], (255,255,255))

# save mask on alpha channel
a_channel = np.where ((b_channel==255), 255,0).astype('uint8')
img = cv2.merge((b_channel, g_channel, r_channel, a_channel))
img = imutils.resize(img, width=1024)

cv2.imwrite('annotated.png', img)