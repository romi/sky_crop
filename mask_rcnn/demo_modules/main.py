import os
import align
import crop
import detect
import assemble
import cv2
import imutils
import numpy as np


class Scan:
    def __init__(self, name):
        self.name = name
        self.dir = '{0}{1}/{1}_aligned_ps.png'.format(folder, self.name)

    def pca_angle(self, filepath):
        self.pca_angle, self.diagonal = align.pca_angle(filepath)
        return self.pca_angle, self.diagonal

    def align(self, filepath, angle, diagonal):
        self.aligned = align.rotate_image(filepath, angle, diagonal)
        return self.aligned

    def crop(self, img):
        self.cropped = crop.crop(img)
        return self.cropped

    def detect(self, img):
        self.detected = detect.detect(img)
        return self.detected

    def assemble(self, points, ind_x, ind_y):
        self.assembled = assemble.assemble_points(points, ind_x, ind_y)
        return self.assembled


def is_date(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    folder = '/Users/soroush/Desktop/Noumena/sky_crop/mask_rcnn/demo/scans/'
    folder_ls = os.listdir(folder)
    dates = []
    for dir in folder_ls:
        if is_date(dir):
            dates.append(dir)
    dates.sort()
    for date in dates:
        a = Scan(date)
        # angle, diagonal = a.pca_angle(a.dir)
        # aligned = a.align(a.dir, angle, diagonal)
        aligned = cv2.imread(a.dir)
        cropped = a.crop(aligned)
        plants_cor = []
        it = 0
        for i in cropped:
            if it < 2:
                cell_x = int(i.split(',')[0])
                cell_y = int(i.split(',')[1])
                img = cropped[i]
                img, cell_points = a.detect(img)
                pts = a.assemble(cell_points, cell_x, cell_y)
                plants_cor = np.append(plants_cor, pts)
                it += 1
            else:
                pass
else:
    pass
