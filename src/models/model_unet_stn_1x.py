# The function implementation below is a modification version from Tensorflow
# Original code link: https://github.com/ashesh6810/DDWP-DA/blob/master/EnKF_DD_all_time.py

import tensorflow
import keras.backend as K

# from data_manager import ClutteredMNIST
# from visualizer import plot_mnist_sample
# from visualizer import print_evaluation
# from visualizer import plot_mnist_grid
import netCDF4
import numpy as np
from keras.layers import (
    Input,
    Convolution2D,
    Convolution1D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    concatenate,
    Activation,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
)
import keras
from keras.callbacks import History


def stn(input_shape=(32, 64, 1), sampling_size=(8, 16), num_classes=10):
    inputs = Input(shape=input_shape)
    conv1 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(inputs)
    conv1 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(pool1)
    conv2 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(pool2)
    # conv3 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv3)

    conv5 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(conv3)
    # conv5 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv5)

    locnet = Flatten()(conv5)
    locnet = Dense(500)(locnet)
    locnet = Activation("relu")(locnet)
    locnet = Dense(200)(locnet)
    locnet = Activation("relu")(locnet)
    locnet = Dense(100)(locnet)
    locnet = Activation("relu")(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation("relu")(locnet)
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights)(locnet)
    x = BilinearInterpolation(sampling_size)([inputs, locnet])

    up6 = keras.layers.Concatenate(axis=-1)(
        [
            Convolution2D(32, 2, 2, activation="relu", border_mode="same")(
                UpSampling2D(size=(2, 2))(x)
            ),
            conv2,
        ]
    )
    conv6 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(up6)
    conv6 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(conv6)

    up7 = keras.layers.Concatenate(axis=-1)(
        [
            Convolution2D(32, 2, 2, activation="relu", border_mode="same")(
                UpSampling2D(size=(2, 2))(conv6)
            ),
            conv1,
        ]
    )
    conv7 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(up7)
    conv7 = Convolution2D(32, 5, 5, activation="relu", border_mode="same")(conv7)

    conv10 = Convolution2D(1, 5, 5, activation="linear", border_mode="same")(conv7)

    model = Model(input=inputs, output=conv10)

    return model
