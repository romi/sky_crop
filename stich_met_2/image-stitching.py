
# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", type=str, required=True,
# 	help="path to input directory of images to stitch")
# ap.add_argument("-o", "--output", type=str, required=True,
# 	help="path to the output image")
# args = vars(ap.parse_args())

# images path
# path = '/Users/aldo/Downloads/image-stitching-opencv/images/romi/'
# images = cv2.imread(path)
# cv2.imshow("test",images)
# output = '/Users/aldo/Downloads/image-stitching-opencv/images/romi/output'

# images = [cv2.imread(file) for file in glob.glob("/Users/aldo/Downloads/image-stitching-opencv/images/romi/'*.png")]
output = '/Users/aldo/Downloads/image-stitching-opencv/images/romi/'

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("/Users/aldo/PycharmProjects/cloud-voxelization/romi/und_imgs")))
print(imagePaths)
images = []

# loop over the image paths, load each one, and add them to our
# images to stitch list
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

# initialize OpenCV's image stitcher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
print(status)
print(stitched)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
    # write the output stitched image to disk
    # cv2.imwrite("output", stitched)
    cv2.imwrite(os.path.join(output, 'stitched.jpg'), stitched)

    # display the output stitched image to our screen
    # cv2.imshow("Stitched", stitched)
    # cv2.waitKey(0)

# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
    print("[INFO] image stitching failed ({})".format(status))