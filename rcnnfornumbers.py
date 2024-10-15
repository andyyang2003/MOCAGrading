import sys
import tensorflow as tf
import cv2
import matplotlib as plt
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from train_mask_rcnn_demo import *
from mrcnn import *

image_path = 'clocks labelled and stuff maybe'
annotation_path = "annotations.jpg"
dataset_train = load_image_dataset(image_path, annotation_path, "train")
dataset_val = load_image_dataset(image_path, annotation_path, "val")
