"""
Michael Patel
June 2020

Project Description:
    To classify whether something is "Manchester United" or not

File description:
    For model preprocessing, training, and inference
"""
################################################################################
# Imports
import os
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf


################################################################################
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3

NUM_EPOCHS = 1
BATCH_SIZE = 64


################################################################################
# Main
if __name__ == "__main__":
    # labels
    classes = []
    int2class = {}
    directories = os.listdir(os.path.join(os.getcwd(), "data"))
    for i in range(len(directories)):
        name = directories[i]
        classes.append(name)
        int2class[i] = name

    num_classes = len(classes)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=50,  # degrees
        width_shift_range=0.9,  # interval [-1.0, 1.0)
        height_shift_range=0.9,  # interval [-1.0, 1.0)
        brightness_range=[0.2, 0.8],  # 0 no brightness, 1 max brightness
        shear_range=0.2,  # stretching in degrees
        zoom_range=[0.5, 1.5],  # less than 1.0 zoom in, more than 1.0 zoom out
        channel_shift_range=175.0,
        # zca_whitening=True,
        # channel_shift_range,
        # horizontal_flip=True,
        # vertical_flip=True,
        rescale=1. / 255  # [0, 255] --> [0, 1]
    )

    train_data_gen = image_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), "data"),
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        color_mode="rgb",
        class_mode="sparse",  # more than 2 classes
        classes=classes,
        batch_size=BATCH_SIZE,
        shuffle=True
        #save_to_dir=os.path.join(os.getcwd(), "x")  # temporary for visualising
    )

    next(train_data_gen)
