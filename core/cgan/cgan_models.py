
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout

import numpy as np


def generator_model_A(input_dimension, channels):

    model = Sequential()

    model.add(Dense(16 * 45 * 45, activation="relu", input_dim=input_dimension))
    model.add(Reshape((45, 45, 16)))
    model.add(UpSampling2D())
    model.add(Conv2D(16, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D())
    model.add(Conv2D(8, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D())
    model.add(Conv2D(4, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(2, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    return model


def discriminator_model_A(input_shape):

    model = Sequential()
    # model.add(Dense(512, input_dim=np.prod(input_shape)))
    model.add(Conv2D(16, (3,3), input_shape=input_shape[:], padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=4, strides=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


