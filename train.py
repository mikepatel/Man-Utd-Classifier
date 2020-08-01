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
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil

import tensorflow as tf


################################################################################
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3

NUM_EPOCHS = 25
BATCH_SIZE = 16


################################################################################
# Main
if __name__ == "__main__":
    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    DATA_DIR = os.path.join(os.getcwd(), "data")
    classes = []
    int2class = {}
    directories = os.listdir(DATA_DIR)
    for i in range(len(directories)):
        name = directories[i]
        classes.append(name)
        int2class[i] = name

    num_classes = len(classes)

    # size of datasets
    num_train_images = 0
    for d in directories:
        images = os.listdir(os.path.join(DATA_DIR, d))
        num_train_images += len(images)

    print(f'Classes: {classes}')
    print(f'Number of classes: {num_classes}')
    print(f'Number of total train images: {num_train_images}')

    # image generator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,  # degrees
        width_shift_range=0.3,  # interval [-1.0, 1.0)
        height_shift_range=0.3,  # interval [-1.0, 1.0),
        brightness_range=[0.5, 1.0],  # 0 is no brightness, 1 is max brightness
        zoom_range=[0.7, 1.3],  # less than 1.0 is zoom in, more than 1.0 is zoom out
        rescale=1./255  # [0, 255] --> [0, 1]
    )

    train_data_gen = image_generator.flow_from_directory(
        directory=DATA_DIR,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=True
        #save_to_dir=os.path.join(os.getcwd(), "temp")
    )

    #next(train_data_gen)

    # ----- MODEL ----- #
    model = tf.keras.Sequential()

    # Convolution 1
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution 2
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution 3
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution 4
    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Flatten
    model.add(tf.keras.layers.Flatten())

    # Fully connected
    model.add(tf.keras.layers.Dense(
        units=128,
        activation=tf.keras.activations.relu
    ))

    # Output
    model.add(tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.sigmoid
    ))

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.summary()

    # ----- TRAIN ----- #
    history = model.fit(
        x=train_data_gen,
        epochs=NUM_EPOCHS,
        steps_per_epoch=num_train_images // BATCH_SIZE
    )

    model.save(os.path.join(os.getcwd(), "saved_model"))

    # plot accuracy
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(os.getcwd(), "training"))

    # ----- TEST ----- #
    TEST_DIR = os.path.join(os.getcwd(), "test")
    test_images = []
    for i in os.listdir(TEST_DIR):
        test_images.append(i)

    for ti in test_images:
        image = Image.open(os.path.join(TEST_DIR, ti))

        image = image.convert("RGB")
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image = np.array(image)
        image = image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
        image = image / 255.0
        image = np.expand_dims(image, 0)

        prediction = model.predict(image)
        prediction = prediction[0][0]
        prediction = int(np.round(prediction))
        prediction = int2class[prediction]
        print(prediction)
        print()

    quit()























    # save model
    #m.save(os.path.join(os.getcwd(), "saved_model"))

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image = np.array(image).astype(np.float) / 255.0
            image = np.expand_dims(image, 0)
            prediction = model.predict(image)
            pred_name = int2class[int(np.argmax(prediction))]
            print(f'{ti}: {pred_name}\n')
