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


def generate_training_data(data, batch_size=32):
    image_batch = np.zeros((batch_size, 66, 220, 3))  # nvidia input params
    label_batch = np.zeros((batch_size))
    while True:
        for i in range(batch_size):
            # generate a random index with a uniform random distribution from 1 to len - 1
            idx = np.random.randint(1, len(data) - 1)

            # Generate a random bright factor to apply to both images
            bright_factor = 0.2 + np.random.uniform()

            row_now = data.iloc[[idx]].reset_index()
            row_prev = data.iloc[[idx - 1]].reset_index()
            row_next = data.iloc[[idx + 1]].reset_index()

            # Find the 3 respective times to determine frame order (current -> next)

            time_now = row_now['image_index'].values[0]
            time_prev = row_prev['image_index'].values[0]
            time_next = row_next['image_index'].values[0]

            if abs(time_now - time_prev) == 1 and time_now > time_prev:
                row1 = row_prev
                row2 = row_now

            elif abs(time_next - time_now) == 1 and time_next > time_now:
                row1 = row_now
                row2 = row_next
            else:
                print('Error generating row')

            x1, y1 = preprocess_image_from_path(
                row1['image_path'].values[0], row1['speed'].values[0], bright_factor)
            # preprocess another image
            x2, y2 = preprocess_image_from_path(
                row2['image_path'].values[0], row2['speed'].values[0], bright_factor)
            # compute optical flow send in images as RGB
            rgb_diff = opticalFlowDense(x1, x2)
            # calculate mean speed
            y = np.mean([y1, y2])
            image_batch[i] = rgb_diff
            label_batch[i] = y

        yield shuffle(image_batch, label_batch)


def generate_validation_data(data):
    while True:
        for idx in range(1, len(data) - 1):
            row_now = data.iloc[[idx]].reset_index()
            row_prev = data.iloc[[idx - 1]].reset_index()
            row_next = data.iloc[[idx + 1]].reset_index()

            # Find the 3 respective times to determine frame order (current -> next)

            time_now = row_now['image_index'].values[0]
            time_prev = row_prev['image_index'].values[0]
            time_next = row_next['image_index'].values[0]

            if abs(time_now - time_prev) == 1 and time_now > time_prev:
                row1 = row_prev
                row2 = row_now

            elif abs(time_next - time_now) == 1 and time_next > time_now:
                row1 = row_now
                row2 = row_next
            else:
                print('Error generating row')

            x1, y1 = preprocess_image_valid_from_path(
                row1['image_path'].values[0], row1['speed'].values[0])
            x2, y2 = preprocess_image_valid_from_path(
                row2['image_path'].values[0], row2['speed'].values[0])

            img_diff = opticalFlowDense(x1, x2)
            img_diff = img_diff.reshape(1, img_diff.shape[0], img_diff.shape[1], img_diff.shape[2])
            y = np.mean([y1, y2])
            speed = np.array([[y]])
            #print('img_diff', img_diff.shape, ' speed', speed)
            yield img_diff, speed
