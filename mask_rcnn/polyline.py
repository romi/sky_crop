# from mrcnn.config import Config
# from mrcnn import model as modellib
# from mrcnn import visualize
# from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import json
import cv2
import os



#read source image
img = cv2.imread('pills/images/35.jpg')
origWidth = img.shape[1]
print(origWidth)

width = 600
image = imutils.resize(img, width)
print(width)

# #visualization
# cv2.imshow('photo',image)
# cv2.waitKey(0)

#list of points
x = [2596, 2547, 2517, 2477, 2376, 2309, 2177, 2079, 2003, 1881, 1768, 1636, 1407, 1312, 1260, 1248, 1275, 1352, 1431, 1541, 1645, 1731, 1832, 1979, 2104, 2254, 2361, 2449, 2514, 2544, 2544, 2541, 2563, 2593, 2608]
y = [789, 789, 816, 896, 1107, 1196, 1352, 1443, 1520, 1596, 1633, 1648, 1673, 1673, 1706, 1755, 1807, 1908, 1954, 2003, 2027, 2034, 2018, 1997, 1951, 1850, 1768, 1651, 1523, 1382, 1242, 1190, 1086, 960, 881]


#ratio
ratio = width / origWidth
print (ratio)

X = [int(i * ratio) for i in x]
print(X)
Y = [int(i * ratio) for i in y]
print(Y)



ptsList  = np.column_stack((X,Y))
#print(ptsList)

pt = (400,400)
r = 80

#cv2.circle(image, (400,350), 20, (0,255,0),-1)
cv2.polylines(image, [ptsList], True, (255, 2550, 255), 5)
cv2.fillPoly(image, [ptsList], 0)
cv2.imshow('region',image)
cv2.waitKey(0)