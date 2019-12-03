import shapely.geometry
import shapely.affinity
from matplotlib import pyplot
from descartes import PolygonPatch
import numpy as np
import pandas as pd
import cv2
import imutils
from mrcnn import visualize

# class RotatedRect:
#     def __init__(self, cx, cy, w, h, angle):
#         self.cx = cx
#         self.cy = cy
#         self.w = w
#         self.h = h
#         self.angle = angle
#
#     def get_contour(self):
#         w = self.w
#         h = self.h
#         c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
#         rc = shapely.affinity.rotate(c, self.angle)
#         return shapely.affinity.translate(rc, self.cx, self.cy)
#
#     def intersection(self, other):
#         print(self.get_contour().intersection(other.get_contour()).area)
#         print(self.get_contour())
#         return self.get_contour().intersection(other.get_contour())
#
#
# r1 = RotatedRect(10, 15, 15, 10, 30)
# r2 = RotatedRect(15, 15, 20, 10, 0)


#####################################################################
#DATA VISUALIZATION



# fig = pyplot.figure(1, figsize=(10, 4))
# ax = fig.add_subplot(121)
# ax.set_xlim(0, 30)
# ax.set_ylim(0, 30)
#
# ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))
# ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))
# ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))
#
# pyplot.show()

#####################################################################
#CREATE MASK FROM DATAFRAME

#read dataframe
dframe = pd.read_csv('detection_df_191411.csv')
mask = dframe.Mask[0]
print(len(mask))
print(mask)

# mask_bool = mask.astype(int)

#open image
imagepath = ('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/examples/LETTUCE_89.JPG')
image = cv2.imread(imagepath)
print(image.shape)
image = imutils.resize(image, width=1024)
print(image.shape)
masks = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
print(len(masks))
ptsList


#
# regionMask = np.zeros(masks.shape[:2], dtype="uint8")
# print(regionMask)
#
# print(mask.dtype, mask.min(), mask.max())
#
# image = visualize.apply_mask(image, mask,(1.0, 0.0, 0.0), alpha=0.4)
# # cv2.imshow('test', image)
# # cv2.waitKey(0)