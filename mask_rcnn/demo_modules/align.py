import cv2
import numpy as np
from sklearn import decomposition
import imutils
from _parameters_ import date, folder, view_process


def pca_angle(filepath):
    print ('filepath\t', filepath)
    im = cv2.imread(filepath, 0)
    print("Image Loaded \t", "height:", im.shape[0], "width:", im.shape[1])
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
    print("PCA Angle\t", angle)
    return angle, diagonal


def rotate_image(filepath, angle, diagonal):
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
    print ('aligned')
    return aligned