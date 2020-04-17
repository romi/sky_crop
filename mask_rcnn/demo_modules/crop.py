import cv2
import os
import glob
import numpy as np
from PIL import Image
from _parameters_ import div_y, div_x, folder, date, view_process


def crop(img):
    height, width = img.shape[0:2]
    limX = int(width / div_x)
    limY = int(height / div_y)
    cropped = {}
    for j in range(div_y):
        for i in range(div_x):
            crop_img = img[limY*j:limY*(j+1), limX*i:limX*(i+1)]
            cropped['{},{}'.format(i,j)] =crop_img
    return cropped