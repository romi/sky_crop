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

#open image
imagepath = ('/Users/aldo/Desktop/git-romi/sky_crop/mask_rcnn/detection/PREDICT_10.png')
pred_img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
pred_img = imutils.resize(pred_img, width=1024)

b,g,r,a = cv2.split(pred_img)
print(a.shape)
# cv2.imshow('alpha',a)
# cv2.waitKey(0)

b = np.zeros(pred_img.shape[:2], dtype="uint8")
b = cv2.circle(b, (500, 400), 200, 255, -1)
print(b.shape)
# cv2.imshow('mask',b)
# cv2.waitKey(0)

print('FIRST IMAGE')
print(a)
print(len(a))
print('SECOND IMAGE')
print(b)
print(len(b))


#CALCULATE INTERSECTION AREA
int_area = np.all([[a == 255],[b == 255]],axis=0)
print(int_area)
int_img = (int_area[0].astype(int))*255
int_img = int_img.astype(np.uint8)
print('INTERSECTION IMAGE')
print(int_img)
print(len(int_img))
cv2.imshow('int', int_img)
cv2.waitKey(0)




