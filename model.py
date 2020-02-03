# libraries to build the model
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import EarlyStopping, ModelCheckpoint


def nvidia_model():
    # model from the paper https://arxiv.org/pdf/1604.07316v1.pdf
    #inputShape = (N_img_height, N_img_width, N_img_channels)
    inputShape = (66, 220, 3)

    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=inputShape))
    # convolution layer 1
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding='valid',
                            kernel_initializer='he_normal', name='conv1'))
    model.add(ELU())

    # convolution layer 2
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding='valid',
                            kernel_initializer='he_normal', name='conv2'))
    model.add(ELU())

    # convolution layer 3
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding='valid',
                            kernel_initializer='he_normal', name='conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    # convolution layer 4
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='valid',
                            kernel_initializer='he_normal', name='conv4'))
    model.add(ELU())

    # convolution layer 5
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='valid',
                            kernel_initializer='he_normal', name='conv5'))
    # flatter for fully connected (FC-layers)
    model.add(Flatten(name='flatten'))
    model.add(ELU())
    # FC layer 1
    model.add(Dense(100, kernel_initializer='he_normal', name='fc1'))
    model.add(ELU())
    # FC layer 2
    model.add(Dense(50, kernel_initializer='he_normal', name='fc2'))
    model.add(ELU())
    # FC layer 3
    model.add(Dense(10, kernel_initializer='he_normal', name='fc3'))
    model.add(ELU())
    # FC layer 4
    model.add(Dense(1, name='output', kernel_initializer='he_normal'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')

    return model
