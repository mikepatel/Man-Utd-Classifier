"""
Michael Patel
June 2020

Project Description:
    To classify whether something is "Manchester United" or not

File description:
    For model inference
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

NUM_EPOCHS = 2000
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

    #model = tf.keras.models.load_model(os.path.join(os.getcwd(), "saved_model"))

    capture = cv2.VideoCapture(0)
    while True:
        # capture frame by frame
        ret, frame = capture.read()

        # preprocess image
        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # crop webcam frame
        y, x, channels = image.shape
        left_x = int(x*0.25)
        right_x = int(x*0.75)
        top_y = int(y*0.25)
        bottom_y = int(y*0.75)
        image = image[top_y:bottom_y, left_x:right_x]

        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        Image.fromarray(image).save(os.path.join(os.getcwd(), "t.png"))

        mod_image = image

        image = np.array(image).astype(np.float32)
        image = image / 255.0
        image = np.expand_dims(image, 0)

        # make prediction
        #prediction = model.predict(image)
        #pred_label = int2class[int(np.argmax(prediction))]
        #print(pred_label)

        # display resulting frame
        cv2.imshow("", mod_image)

        if cv2.waitKey(1) == 27:  # continuous stream, escape key
            break

        # release capture
    capture.release()
    cv2.destroyAllWindows()
