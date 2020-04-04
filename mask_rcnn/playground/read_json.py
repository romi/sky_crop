import cv2
import numpy as np
import glob
import json

#PARAMETERS
path_img = '/Users/aldo/Desktop/untitled/01-training/images/0.JPG'
folder_imgs = '/Users/aldo/Desktop/untitled/01-training/images/'
json_path = '/Users/aldo/Desktop/git_romi/sky_crop/mask_rcnn/playground/via_region_data.json'
filenames = [img for img in glob.glob("/Users/aldo/Desktop/untitled/01-training/images/*.JPG")]

org = filenames.sort()
print(org)

# read image
img = cv2.imread(path_img,1)
height,width,depth = img.shape

#open file
with open(json_path) as json_file:
    data = json.load(json_file)

    for item in data:
        mask = np.zeros((height, width), dtype="uint8")
        print(data[item]['filename'])

        # draw polylines
        points_x = data[item]['regions'][0]['shape_attributes']['all_points_x']
        points_y = data[item]['regions'][0]['shape_attributes']['all_points_y']
        ptsList = np.column_stack((points_x, points_y))
        cv2.polylines(mask, [ptsList], True, (255, 2550, 255), 10)

    for filename in os.listdir(folder_imgs):
        print(filename)
        cv2.imread()


        # mask_resized = imutils.resize(mask, width=800)
        # cv2.imshow('mask-polyline',mask_resized)
        # cv2.waitKey(0)
