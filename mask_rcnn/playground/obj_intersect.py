import os
import json
import cv2
import imutils
import numpy as np


abs_path = os.path.abspath(os.path.dirname(__file__))
path = (os.path.join(abs_path, "../IoU/via_export_json.json"))

def parse_JSON ():
    pts_list = {}
    with open(path, 'r') as myfile:
        data = myfile.read()
    json_data = json.loads(data)
    for i in json_data:
        if i[0] != "_":
            file = (json_data[i]['filename'])
            x = (json_data[i]['regions'][0]['shape_attributes']['all_points_x'])
            y = (json_data[i]['regions'][0]['shape_attributes']['all_points_y'])
            pts_list[file] = np.column_stack((x,y))
    return pts_list

def draw_mask(image_path, ptsList):  # to draw mask from polyline on the raw image
    # read image
    img = cv2.imread(image_path)

    # draw white mask on black background
    background = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # draw and fill polyline
    bkg = cv2.fillPoly(background, [ptsList], (255, 255, 255))
    # add mask in alpha channel
    image_b = imutils.resize(bkg, width=1024)

    b, g, image_b = cv2.split(image_b)
    print ("Mask_IoU Image")
    # cv2.imshow('mask_IoU', pline)
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

    print ("INTERSECTION AREA: ", np.sum(int_area[0].astype(int)))

    # cv2.imshow('int', int_img)
    # cv2.waitKey(0)
    return int_img


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
    pline = cv2.polylines(output, np.int32([pts_list]), True, (0, 220, 220), 6)
    output = imutils.resize(output,width=1024)
    overlay = calculate_int_area(image_a,image_b)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.imshow(i, output)
    cv2.waitKey(0)