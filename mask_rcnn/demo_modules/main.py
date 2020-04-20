import os
import align
import crop
import detect
import assemble
import cv2
import imutils
import copy
import numpy as np
import pandas as pd
from PIL import Image


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

    def assemble_points(self, points, ind_x, ind_y):
        self.plant_coordinates = assemble.assemble_points(points, ind_x, ind_y)
        return self.plant_coordinates

    # def detected_img(self, img):
        # return self.detected_crop


def is_date(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    folder = '/Users/soroush/Desktop/Noumena/sky_crop_soroush/mask_rcnn/demo/scans/'
    folder_ls = os.listdir(folder)
    dates = []
    for dir in folder_ls:
        if is_date(dir):
            dates.append(dir)
    dates.sort()
    data = {}
    tt = 0

    # ------------------ for dataframe -------------------------
    # scans = {}
    # plants = []
    # names = []
    # for i in range(20):
    #     plants.append('{}:area'.format(i))
    #     names.append('p-{}'.format(i))
    # scans['plant'] = names
    # for date in dates:
    #     scans[date] = plants
    # dataframe = pd.DataFrame(data=scans, columns=scans.keys())
    # print(dataframe)
    
    for date in dates:
        if tt < 1:
            tt += 1
            plants = []
            for i in range(20):
                plants.append(i)
            a = Scan(date)
            # angle, diagonal = a.pca_angle(a.dir)
            # aligned = a.align(a.dir, angle, diagonal)
            aligned = cv2.imread(a.dir)
            height, width = aligned.shape[:2]
            cropped = a.crop(aligned)
            plants_cor = np.zeros((1, 2), dtype=np.int32)
            total_detected = copy.copy(aligned)
            t = 0
            for i in cropped:
                # only runs the code for first cropped image, remove this "if" part to get the full detected image
                if t < 2:
                    t += 1
                    cell_x = int(i.split(',')[0])
                    cell_y = int(i.split(',')[1])
                    img = cropped[i]
                    img, cell_points = a.detect(img)
                    x_offset = img.shape[1] * cell_x
                    y_offset = img.shape[0] * cell_y
                    total_detected[y_offset:y_offset+img.shape[0],
                                   x_offset:x_offset+img.shape[1]] = img
                    pts = a.assemble_points(cell_points, cell_x, cell_y)
                    plants_cor = np.append(plants_cor, pts, axis=0)
            plants_cor = plants_cor[1:, :]
            total_detected = imutils.resize(total_detected, width=1200)
            a.detected_crop = total_detected
            cv2.imshow('total_detected', a.detected_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            data[a.name] = [plants_cor]
    dataframe = pd.DataFrame(data, columns=dates)
    print (dataframe)
    
else:
    pass
