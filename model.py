# import libraries
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, ELU, TimeDistributed, Flatten, Dropout, Lambda
from keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D

IMG_SHAPE = (66, 220, 3)


def nvidia_model():

    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=IMG_SHAPE))
    # conv layer 1
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid',
                     kernel_initializer='he_normal', name='conv1'))
    model.add(ELU())
    # conv layer 2
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid',
                     kernel_initializer='he_normal', name='conv2'))
    model.add(ELU())
    # conv layer 3
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid',
                     kernel_initializer='he_normal', name='conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    # conv layer 4
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
                     kernel_initializer='he_normal', name='conv4'))
    model.add(ELU())
    # conv layer 5
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
                     kernel_initializer='he_normal', name='conv5'))

    model.add(Flatten(name='flatten'))
    model.add(ELU())
    # fc layer 1
    model.add(Dense(100, kernel_initializer='he_normal', name='fc1'))
    model.add(ELU())
    # fc layer 2
    model.add(Dense(50, kernel_initializer='he_normal', name='fc2'))
    model.add(ELU())
    # fc layer 3
    model.add(Dense(10, kernel_initializer='he_normal', name='fc3'))
    model.add(ELU())
    # final/output layer
    model.add(Dense(1, name='output', kernel_initializer='he_normal'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='mse')

    return model


if __name__ == "__main__":
    print("These are models from Nvidia end to end paper")
