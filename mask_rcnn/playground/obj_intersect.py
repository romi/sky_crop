from imutils import paths
import numpy as np
import argparse
import imutils
import random
import json
import cv2
import os

path = "/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/IoU/via_project_6Dec2019_15h1m.json"

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
img = cv2.imread('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/IoU/10.JPG')
b_channel, g_channel, r_channel = cv2.split(img)

# draw white mask on black background
background = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
ptsList  = np.column_stack((x,y))
bkg=cv2.polylines(background, [ptsList], True, (0, 0, 0), 1)
bkg=cv2.fillPoly(background, [ptsList], (255,255,255))

# # save mask on alpha channel
# a_channel = np.where ((b_channel==255), 255,0).astype('uint8')
# img = cv2.merge((b_channel, g_channel, r_channel, a_channel))
image_b = imutils.resize(bkg, width=1024)
b,g,image_b= cv2.split(image_b)
print(image_b.shape)
cv2.imshow('mask_IoU',image_b)


#open predicted image
imagepath = ('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/detection/PREDICT_10.png')
pred_img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
pred_img = imutils.resize(pred_img, width=1024)

b,g,r,image_a = cv2.split(pred_img)
print(image_a.shape)
cv2.imshow('predicted', image_a)



#CALCULATE INTERSECTION AREA
# int_area = np.all([[image_b == 255],[image_a == 255]],axis=0)
int_area = np.logical_xor([image_b == 255],[image_a == 255])
print(int_area)
int_img = (int_area[0].astype(int))*255
int_img = int_img.astype(np.uint8)
print('INTERSECTION IMAGE')
print(int_img)
print(len(int_img))
cv2.imshow('int', int_img)
cv2.waitKey(0)