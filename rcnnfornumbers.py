import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import tensorflow as tf
from mrcnn import model as modellib


from mrcnn.config import CONFIG
from mrcnn import utils

image_path = 'clocks labelled and stuff maybe'
annotation_path = "annotations.jpg"

class DigitConfig(COCO.CocoConfig):
    NAME = "digit_segmentation"
    IMAGES_PER_GPU = 1  # Adjust based on your GPU
    NUM_CLASSES = 1 + 10  # Background + digits 0-9
    STEPS_PER_EPOCH = 100  # Adjust based on dataset size
    DETECTION_MIN_CONFIDENCE = 0.9
class DigitDataset(utils.Dataset):
    def load_digits(self, dataset_dir, subset):
        # Initialize COCO API for instance annotations
        coco = COCO("{}/annotations/instances_{}.json".format(dataset_dir, subset))
        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())

        # Add digit classes
        for i in range(10):  # Assuming digit labels from 0 to 9
            self.add_class("digits", i + 1, str(i))

        # Add images and their annotations
        for image_id in image_ids:
            self.add_image(
                "digits",
                image_id=image_id,
                path=os.path.join(dataset_dir, coco.imgs[image_id]['file_name']),
                width=coco.imgs[image_id]["width"],
                height=coco.imgs[image_id]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=class_ids))
            )

    def load_mask(self, image_id):
        # Load instance masks for the image
        info = self.image_info[image_id]
        annotations = info['annotations']

        masks = []
        class_ids = []
        for annotation in annotations:
            mask = COCO.annToMask(annotation).astype(bool)
            masks.append(mask)
            class_ids.append(annotation['category_id'])

        # Convert to a 3D array with a mask per instance
        masks = np.stack(masks, axis=-1) if masks else np.empty((0, 0, 0))
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return masks, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]
# Load the training and validation datasets
dataset_train = DigitDataset()
dataset_train.load_digits("clocks labelled and stuff maybe", "train")
dataset_train.prepare()

dataset_val = DigitDataset()
dataset_val.load_digits("clocks labelled and stuff maybe", "val")
dataset_val.prepare()

# Configure the model for training
config = DigitConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir="./") 
# Load weights - either COCO weights or pretrained weights from your own model
COCO_WEIGHTS_PATH = "path/to/mask_rcnn_coco.h5"  # Change if using pretrained weights
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                            "mrcnn_bbox", "mrcnn_mask"])

# Train the model
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers="heads")  # You can also specify "all" to fine-tune the whole model