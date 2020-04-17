from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import json
import cv2
import os
from _parameters_ import date, folder, LOGS_AND_MODEL_DIR


CLASS_NAMES = {1: "lettuce"}


class ObjConfig(Config):
    # give the configuration a recognizable name
    NAME = "lettuce"

    # set the number of GPUs to use training along with the number of
    # images per GPU (which may have to be tuned depending on how
    # much memory your GPU has)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # set the number of steps per training epoch
    STEPS_PER_EPOCH = 1

    # number of classes (+1 for the background)
    NUM_CLASSES = len(CLASS_NAMES) + 1


class ObjInferenceConfig(ObjConfig):
    # set the number of GPUs and images per GPU (which may be
    # different values than the ones used for training)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # set the minimum detection confidence (used to prune out false
    # positive detections)
    DETECTION_MIN_CONFIDENCE = 0.9


def detect(image):
    # initialize the inference configuration
    config = ObjInferenceConfig()

    # initialize the Mask R-CNN model for inference
    model = modellib.MaskRCNN(
        mode="inference", config=config, model_dir=LOGS_AND_MODEL_DIR)

    # load our trained Mask R-CNN
    weights = model.find_last()
    model.load_weights(weights, by_name=True)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = model.detect([image], verbose=1)[0]

    for i in range(r["rois"].shape[0]):
        mask = r["masks"][:, :, i]
        unique, counts = np.unique(mask, return_counts=True)
        mask_items = dict(zip(unique, counts))
        ratio = mask_items[True] / (mask_items[True] + mask_items[False])
        image = visualize.apply_mask(image, mask, (1.0, 0.0, 0.0), alpha=0.4)
        image = visualize.draw_box(image, r["rois"][i], (1.0, 0.0, 0.0))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # loop over the predicted scores and class labels
    plants = []
    for ind, i in enumerate(range(len(r["scores"]))):

        # extract the bounding box information, class ID, label,
        # and predicted probability from the results
        (startY, startX, endY, endX) = r["rois"][i]
        classID = r["class_ids"][i]
        label = CLASS_NAMES[classID]
        score = r["scores"][i]
        mid_point = [(startX+endX)//2, (startY+endY)//2]
        center_point = {"X": str(mid_point[0]), "Y": str(mid_point[1]), 
                "range_x": str(image.shape[1]), "range_y": str(image.shape[0])}
        plants.append(center_point)
        # draw the class label and score on the image
        text="{}: {:.4f}".format(label, score)
        y=startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # image=imutils.resize(image, width=1000)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    return image, plants