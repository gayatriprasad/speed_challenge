import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split

import h5py

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import nvidia_model
from helper_functions import train_valid_split, change_brightness, opticalFlowDense, preprocess_image, preprocess_image_valid_from_path, preprocess_image_from_path
from generator_dataset import generate_training_data, generate_validation_data

# constants
DATA_PATH = '/home/saiperi/speedchallenge/data2'
TRAIN_VIDEO = os.path.join(DATA_PATH, 'train.mp4')
TEST_VIDEO = os.path.join(DATA_PATH, 'test.mp4')
CLEAN_DATA_PATH = '/home/saiperi/speedchallenge/data2/clean_data'
# if Path to raw image folder does not exists make folder
if not os.path.exists(CLEAN_DATA_PATH):
    os.makedirs(CLEAN_DATA_PATH)
CLEAN_IMGS_TRAIN = os.path.join(CLEAN_DATA_PATH, 'train_imgs')
if not os.path.exists(CLEAN_IMGS_TRAIN):
    os.makedirs(CLEAN_IMGS_TRAIN)
CLEAN_IMGS_TEST = os.path.join(CLEAN_DATA_PATH, 'test_imgs')
if not os.path.exists(CLEAN_IMGS_TEST):
    os.makedirs(CLEAN_IMGS_TEST)

# number of frames
train_frames = 15057
test_frames = 14496

seeds = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


train_meta = pd.read_csv(os.path.join(CLEAN_DATA_PATH, 'train_meta.csv'))
print('shape: ', train_meta.shape)


val_size = len(valid_data.index)
valid_generator = generate_validation_data(valid_data)
BATCH = 16
print('val_size: ', val_size)

filepath = 'model-weights-Vtest3.h5'
earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=1,
                              verbose=1,
                              min_delta=0.23,
                              mode='min',)
modelCheckpoint = ModelCheckpoint(filepath,
                                  monitor='val_loss',
                                  save_best_only=True,
                                  mode='min',
                                  verbose=1,
                                  save_weights_only=True)
callbacks_list = [modelCheckpoint]


model = nvidia_model()
train_size = len(train_data.index)
train_generator = generate_training_data(train_data, BATCH)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=400,
    epochs=85,
    callbacks=callbacks_list,
    verbose=1,
    validation_data=valid_generator,
    validation_steps=val_size)

print(history)

# plot the training and validation loss for each epoch
fig, ax = plt.subplots(figsize=(20, 10))
plt.plot(history.history['loss'], 'ro--')
plt.plot(history.history['val_loss'], 'go--')
plt.title('Model-v2test mean squared error loss 15 epochs')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('/home/saiperi/speedchallenge/data2/clean_data/MSE_per_epoch.png')
plt.close()

print('done')
