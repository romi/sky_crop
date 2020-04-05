import cv2
import numpy as np
import imutils
import json
import os

#PARAMETERS
path = '/Users/aldo/Desktop/untitled/01-training/images/'
json_path = '/Users/aldo/Desktop/git_romi/sky_crop/mask_rcnn/playground/via_region_data.json'

#open file
with open(json_path) as json_file:
    data = json.load(json_file)

    for item in data:
        img_name = data[item]['filename']
        img_path = os.path.join(path,img_name)
        print(img_path)

        # read image
        img = cv2.imread(img_path, 1)
        height, width, depth = img.shape

        # create alpha channel
        # mask = np.zeros((height, width), dtype="uint8"     )
        overlay = img.copy()
        alpha = 0.4  # Transparency factor.

        # draw polylines
        points_x = data[item]['regions'][0]['shape_attributes']['all_points_x']
        points_y = data[item]['regions'][0]['shape_attributes']['all_points_y']
        ptsList = np.column_stack((points_x, points_y))
        cv2.polylines(overlay, [ptsList], True, (255, 2550, 255), 20)
        cv2.fillPoly(overlay, [ptsList], 255)
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # visualize image
        img_resized = imutils.resize(image_new, width=800)
        cv2.imshow(img_name,img_resized)
        cv2.waitKey(0)

        # # mask on image
        # masked_data = cv2.bitwise_and(img, img, mask=mask_resized)

# https://stackoverflow.com/questions/25074488/how-to-mask-an-image-using-numpy-opencv
