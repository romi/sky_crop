import cv2
import numpy as np
from sklearn import decomposition
import imutils
from _parameters_ import date,folder, view_process

filepath = '{0}/{1}_orthomosaic.tif'.format(folder, date)

im = cv2.imread(filepath, 0)
print("-------------------------------------------")
print("Starting Align Orthomosaic\n")
print("IMAGE LOADED \t", "height:", im.shape[0], "width:", im.shape[1])
idxs = np.array(np.where(im)).T

p = decomposition.PCA()
res = p.fit(idxs)
tidxs = res.transform(idxs)

width = np.int(tidxs[:, 0].max() - tidxs[:, 0].min())
height = np.int(tidxs[:, 1].max() - tidxs[:, 1].min())
diagonal = max(width, height)

x_axis = res.components_[0][0]
y_axis = res.components_[0][1]

angle = np.arctan(x_axis / y_axis)
print ("PCA Angle:\t", angle)

if angle < (-np.pi/2):
    angle = angle - np.pi
elif angle > 0 and angle < np.pi/2:
    angle = angle + np.pi

im = cv2.imread(filepath, 3)
rot_mat = cv2.getRotationMatrix2D(
    (im.shape[0], im.shape[1]), angle * 180 / np.pi, 1.0)
warped = cv2.warpAffine(
    im, rot_mat, (diagonal*2, diagonal), flags=cv2.INTER_LINEAR)
idxs = np.array(np.where(warped[:, :, 0])).T
minx, miny = idxs.min(axis=0)
maxx, maxy = idxs.max(axis=0)
aligned = warped[minx:maxx, miny:maxy]

cv2.imwrite('{0}/{1}_aligned.png'.format(folder,date), aligned)

if view_process:
    cv2.imshow("Aligned Image", imutils.resize(aligned,1000))
    cv2.waitKey(0)

print ("Aligned\t\t", "file saved!")
print("-------------------------------------------")

# # PLOT DATA
# plt.scatter(idxs[:, 0], idxs[:, 1], s=0.05, alpha=0.01)
# for length, vector in zip(p.explained_variance_, p.components_):
#     v = vector * 1 * np.sqrt(length)
#     draw_vector(p.mean_, p.mean_ + v)
# plt.axis('equal')
# plt.show()