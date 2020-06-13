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
        class_mode="binary",  # more than 2 classes
        classes=classes,
        batch_size=BATCH_SIZE,
        shuffle=True
        #save_to_dir=os.path.join(os.getcwd(), "x")  # temporary for visualising
    )

    next(train_data_gen)

    m = tf.keras.Sequential()

    # ----- Stage 1 ----- #
    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Max Pooling
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    # Dropout
    m.add(tf.keras.layers.Dropout(
        rate=0.2
    ))

    # ----- Stage 2 ----- #
    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Max Pooling
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    # Dropout
    m.add(tf.keras.layers.Dropout(
        rate=0.2
    ))

    # ----- Stage 3 ----- #
    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Max Pooling
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    # Dropout
    m.add(tf.keras.layers.Dropout(
        rate=0.2
    ))

    # ----- Stage 4 ----- #
    # Flatten
    m.add(tf.keras.layers.Flatten())

    # Dense
    m.add(tf.keras.layers.Dense(
        units=256,
        activation=tf.keras.activations.relu
    ))

    # Dropout
    m.add(tf.keras.layers.Dropout(
        rate=0.5
    ))

    # Dense - output
    m.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    ))

    m.compile(
        loss=tf.keras.losses.BinaryCrossentropy,
        optimmizer=tf.keras.optimizers.Adam(0.0001),
        metrics=["accuracy"]
    )

    m.summary()

    history = m.fit(
        x=train_data_gen,
        epochs=NUM_EPOCHS
    )

    m.save(os.path.join(os.getcwd(), "saved_model"))

    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "saved_model"))

    image = cv2.imread(os.path.join(os.getcwd(), "test\\manc.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = np.array(image).astype(np.float) / 255.0
    image = np.expand_dims(image, 0)
    prediction = model.predict(image)
    print(int2class[int(np.argmax(prediction))])
