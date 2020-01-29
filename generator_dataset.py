# import libraries
from video_to_images_dataset_conversion import FrameSpeedDataset
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import os
from utility_functions import change_brightness, opticalFlowDense, preprocess_image, preprocess_image_valid_from_path, preprocess_image_from_path

"""
data_dir = '/home/gp/Documents/projects/speed-challenge-2017/data'
# if Path to raw image folder does not exists make folder
path_to_images = os.path.join(data_dir + '/IMG')
if not os.path.exists(path_to_images):
    os.makedirs(path_to_images)
# reading the train speeds
frame_speed_df = pd.read_csv(os.path.join(data_dir, 'train.txt'), header=None, squeeze=True)
frame_speed_df = pd.DataFrame(
    {'Frame': frame_speed_df.index, 'Speed': frame_speed_df.values})
fsd = FrameSpeedDataset(data_dir, path_to_images, frame_speed_df)
"""


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
            time_now = row_now['image_path'].values[0]
            time_prev = row_prev['image_path'].values[0]
            time_next = row_next['image_path'].values[0]

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
        #print('image_batch', image_batch.shape, ' label_batch', label_batch)
        # Shuffle the pairs before they get fed into the network
        yield shuffle(image_batch, label_batch)


def generate_validation_data(data):
    while True:
        # start from the second row because we may try to grab it and need its prev to be in bounds
        for idx in range(1, len(data) - 1):
            row_now = data.iloc[[idx]].reset_index()
            row_prev = data.iloc[[idx - 1]].reset_index()
            row_next = data.iloc[[idx + 1]].reset_index()
            # Find the 3 respective times to determine frame order (current -> next)
            time_now = row_now['image_path'].values[0]
            time_prev = row_prev['image_path'].values[0]
            time_next = row_next['image_path'].values[0]

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


if __name__ == "__main__":
    pass
    """
    filepath = 'model-weights.h5'
    df = pd.read_csv("./processed.csv", header=None)
    pre = PreProcessor()
    train, test = fsd.shuffle_frame_pairs(df)
    size_test = len(test.index)
    size_train = len(train.index)
    # print(size_test)
    # print(size_train)
    dl_model = model.speed_model()
    earlyStopping = EarlyStopping(monitor='val_loss',patience=2,verbose=1,min_delta=0.23,mode='min',)
    modelCheckpoint = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True,mode='min',verbose=1,save_weights_only=True)
    callbacks_list = [modelCheckpoint, earlyStopping]
    train_generator = generate_training_data(train)
    test_generator = generate_validation_data(test)
    history = dl_model.fit_generator(train_generator,steps_per_epoch=555,epochs=25,callbacks=callbacks_list,verbose=1,validation_data=test_generator,validation_steps=size_test)
    print(history)
    """
