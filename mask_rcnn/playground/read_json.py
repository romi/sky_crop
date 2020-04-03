import cv2
import numpy as np
import imutils
import random
import json

#PARAMETERS
path = '/Users/aldo/Desktop/untitled/01-training/images/0.JPG'
json_path = '/Users/aldo/Desktop/git_romi/sky_crop/mask_rcnn/playground/via_region_data.json'

# read image
img = cv2.imread(path,1)

scale_percent = 10 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow('test', resized)
# cv2.waitKey(0)

#open file
with open(json_path) as json_file:
    data = json.load(json_file)
    print(data)
#     for p in data['people']:
#         print('Name: ' + p['name'])
#         print('Website: ' + p['website'])
#         print('From: ' + p['from'])
#         print('')


# # allocate memory for our [height, width, num_instances] array
# # where each "instance" effectively has its own "channel"
# masks = np.zeros((img.shape[0], img.shape[1],
#                   len(annot["regions"])), dtype="uint8")
#
# # loop over each of the annotated regions
# for (i, region) in enumerate(annot["regions"]):
#     # allocate memory for the region mask
#     regionMask = np.zeros(masks.shape[:2], dtype="uint8")
#
#     # grab the shape and region attributes
#     sa = region["shape_attributes"]
#     ra = region["region_attributes"]
#
#     # scale the center (x, y)-coordinates and radius of the
#     # circle based on the dimensions of the resized image
#     ratio = info["width"] / float(info["orig_width"])
#
#     X = [int(i * ratio) for i in sa["all_points_x"]]
#     Y = [int(i * ratio) for i in sa["all_points_y"]]
#     ptsList = np.column_stack((X, Y))
#
#     # r = int(sa["r"] * ratio)
#
#     # draw a circular mask for the region and store the mask
#     # in the masks array
#     # cv2.circle(regionMask, (cX, cY), r, 1, -1)
#     cv2.polylines(regionMask, [ptsList], True, (255, 2550, 255), 10)
#     cv2.fillPoly(regionMask, [ptsList], 255)
#     masks[:, :, i] = regionMask

