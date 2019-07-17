from imgaug import augmenters as iaa
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os

# initialize the dataset path, images path, and annotations file path
DATASET_PATH = os.path.abspath("isic2018")
IMAGES_PATH = os.path.sep.join([DATASET_PATH,
     "ISIC2018_Task1-2_Training_Input"])
MASKS_PATH = os.path.sep.join([DATASET_PATH,
     "ISIC2018_Task1_Training_GroundTruth"])

# initialize the amount of data to use for training
TRAINING_SPLIT = 0.8


# grab all image paths, then randomly select indexes for both training
# and validation
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))
idxs = list(range(0, len(IMAGE_PATHS)))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)
trainIdxs = idxs[:i]
valIdxs = idxs[i:]

# initialize the class names dictionary
CLASS_NAMES = {1: "lesion"}

# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "mask_rcnn_coco.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = "lesions_logs"