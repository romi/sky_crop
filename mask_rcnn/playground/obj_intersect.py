import os
import json
import cv2
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

abs_path = os.path.abspath(os.path.dirname(__file__))
path = (os.path.join(abs_path, "../IoU/via_export_json.json"))

def parse_JSON ():
    pts_list = {}
    with open(path, 'r') as myfile:
        data = myfile.read()
    json_data = json.loads(data)
    for i in json_data:
        if i[0] != "_":
            pts = []
            file = (json_data[i]['filename'])
            regions = json_data[i]['regions']
            for k in range(len(regions)):
                x = (regions[k]['shape_attributes']['all_points_x'])
                y = (regions[k]['shape_attributes']['all_points_y'])
                pts.append(np.column_stack((x,y)))
            pts_list[file] = pts
    return pts_list

def draw_mask(image_path, ptsList):  # to draw mask from polyline on the raw image
    # read image
    img = cv2.imread(image_path)
    background = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)        # draw white mask on black background
    for j in ptsList:
        bkg = cv2.fillPoly(background, [j], (255, 255, 255)) # draw and fill polyline
    # add mask in alpha channel
    image_b = imutils.resize(bkg, width=1024)

    b, g, image_b = cv2.split(image_b)
    print ("Mask_IoU Image")

    # cv2.imshow('mask_IoU', image_b)
    # cv2.waitKey(0)
    # cv2.destroyWindow('mask_IoU')
    return image_b

def load_predicted(imagepath):
    pred_img = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
    pred_img = imutils.resize(pred_img, width=1024)
    b, g, r, image_a = cv2.split(pred_img)
    print("Predicted Image")

    # cv2.imshow('predicted', image_a)
    # cv2.waitKey(0)
    # cv2.destroyWindow('predicted')
    return image_a

def calculate_int_area (image_a, image_b):
    int_area = np.logical_xor([image_b == 255], [image_a == 255])
    int_img = (int_area[0].astype(int)) * 255
    int_img = int_img.astype(np.uint8)
    area = np.sum(int_area[0].astype(int))

    print ("INTERSECTION AREA: ",area )

    # cv2.imshow('int', int_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('int')
    return int_img, area

#list of overlapping area
calc_area = []

for i in parse_JSON():
    # read image name from json file
    image_b_path = (os.path.join(abs_path, "../examples/",i))
    print ("IMAGE: ", i)

    # draw mask for the image
    pts_list = parse_JSON()[i] # list of points for polyline
    image_b = draw_mask(image_b_path,pts_list)

    # get the mask from predicted image
    suffix =  "../detection/PREDICT_" + i[:-3] + "png"
    image_a_path = (os.path.join(abs_path, suffix))
    image_a = load_predicted(image_a_path)

    # compare these two masks
    output = cv2.imread(image_b_path)

    # pline = cv2.polylines(output, np.int32([pts_list]), True, (0, 220, 220), 6)
    output = imutils.resize(output,width=1024)
    overlay, area= calculate_int_area(image_a,image_b)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    alpha = 0.5
    calc_area.append(area)
    # cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # cv2.imshow(i, output)
    # cv2.waitKey(0)

print(calc_area)

# add column to dataframe
csv_path = (os.path.join(abs_path, "../detection_df.csv"))
data = pd.read_csv(csv_path)
print(data)
data = data.groupby(['Log','Img_Name'],as_index=False)['Scores'].mean()
data['Inter_Area'] = calc_area
print(data)

# Explanatory Data Analysis (EDA)
# Reference_https://www.youtube.com/watch?v=3r62Gt7-hVs&list=RD8aEAAIm-oz8&index=2, https://vita.had.co.nz/papers/gpp.pdf
# pairs plot
x = data['Inter_Area'].tolist()
y = data['Scores'].tolist()

print(x,y)

sns.set_style("white")
dataplot = sns.lmplot('Scores','Inter_Area',data,
                      height=7, aspect=0.9,
                      scatter_kws=dict(s=30, linewidths=.5),fit_reg=False)
dataplot.set(xlim=(0.965, 1.0), ylim=(100, 40000))
plt.show()