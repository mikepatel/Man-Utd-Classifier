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
IMAGE_CHANNELS = 4

NUM_EPOCHS = 500
BATCH_SIZE = 128


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

    """
    # convert and save images to rgba
    q = Image.open(os.path.join(os.getcwd(), "data\\Man United\\manchester-united-fc.jpg"))
    q = q.convert("RGBA")
    background = Image.new("RGBA", q.size, (255, 255, 255))
    q = Image.alpha_composite(background, q)
    q = q.convert("RGB")
    q.show()
    quit()
    """

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,  # degrees
        width_shift_range=0.3,  # interval [-1.0, 1.0)
        height_shift_range=0.3,  # interval [-1.0, 1.0)
        brightness_range=[0.5, 1.0],  # 0 no brightness, 1 max brightness
        shear_range=20,  # stretching in degrees
        zoom_range=[0.7, 1.3],  # less than 1.0 zoom in, more than 1.0 zoom out
        #channel_shift_range=100.0,
        #zca_whitening=True,
        # horizontal_flip=True,
        # vertical_flip=True,
        rescale=1. / 255  # [0, 255] --> [0, 1]
    )

    train_data_gen = image_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), "data"),
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        #color_mode="rgb",
        color_mode="rgba",
        class_mode="binary",  # more than 2 classes
        classes=classes,
        batch_size=BATCH_SIZE,
        shuffle=True,
        save_to_dir=os.path.join(os.getcwd(), "x")  # temporary for visualising
    )

    #x = next(train_data_gen)
    #quit()

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
        activation=tf.keras.activations.sigmoid
    ))

    m.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimmizer=tf.keras.optimizers.Adam(0.0001),
        metrics=["accuracy"]
    )

    m.summary()

    history = m.fit(
        x=train_data_gen,
        epochs=NUM_EPOCHS
    )

    # save model
    m.save(os.path.join(os.getcwd(), "saved_model"))

    # ----- TEST ----- #
    # load model
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), "saved_model"))

    test_images = []
    for i in os.listdir(os.path.join(os.getcwd(), "test")):
        test_images.append(i)

    """
    test_images = [
        "manc.png",
        "manc_cropped.jpg",
        "mancw.jpg"
    ]
    """

    for i in range(3):
        for ti in test_images:
            """
            image = Image.open(os.path.join(os.getcwd(), "test\\"+ti))
            image = image.convert("RGBA")
            #image.show()
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            image = np.array(image).astype(np.float32)
            image = image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
            image = image / 255.0
            image = np.expand_dims(image, 0)
            prediction = model.predict(image)
            pred_name = int2class[int(np.argmax(prediction))]
            print(f'{ti}: {pred_name}\n')
            """

            image = cv2.imread(os.path.join(os.getcwd(), "test\\"+ti))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = np.array(image).astype(np.float) / 255.0
            image = np.expand_dims(image, 0)
            prediction = model.predict(image)
            pred_name = int2class[int(np.argmax(prediction))]
            print(f'{ti}: {pred_name}\n')
