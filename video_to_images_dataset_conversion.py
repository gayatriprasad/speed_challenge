# import libraries
import numpy as np
import cv2
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import random
from utility_functions import change_brightness, opticalFlowDense, preprocess_image, preprocess_image_valid_from_path, preprocess_image_from_path


class FrameSpeedDataset:
    """ Returns data, speeds
        The shape of data is 3 x 320 x 70
        The speeds are of type float
    """

    def __init__(self, train_data_path, path_to_images, train_speed_df):
        self.train_data_path = train_data_path
        self.path_to_images = path_to_images
        self.train_speed_df = train_speed_df

    def __str__(self):
        return 'https://github.com/commaai/speedchallenge'

    def generate_images_from_video(self):
        # plot the speed at each frames
        frame = np.asarray(self.train_speed_df['Frame'], dtype=np.float32)
        print('Number of frames:', len(frame))
        speed = np.asarray(self.train_speed_df['Speed'], dtype=np.float32)
        print()
        """
        # test to plot the figure
        plt.plot(frame, speed, 'r-')
        plt.title('Speed vs frame')
        plt.show()
        """
        # construct a dastaset by writing to a csv file the image path and speed
        with open(self.train_data_path + '/driving_log.csv', 'w') as csvfile:
            fieldnames = ['image_path', 'frame', 'speed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # open video
            vidcap = cv2.VideoCapture(os.path.join(self.train_data_path, 'train.mp4'))
            success, image = vidcap.read()
            count = 0
            while success:
                # save images in a soecific folder
                os.chdir(self.path_to_images)
                # saving images
                cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
                success, image = vidcap.read()
                print('Read a new frame: ', success)
                # write row to driving.csv
                writer.writerow({'image_path': self.path_to_images,
                                 'frame': count, 'speed': self.train_speed_df['Speed'][count]})
                image_path = os.path.join(self.path_to_images, str(count) + '.jpg')
                count += 1

        print('Video to Images converstion done!')

        # shuffle frame pairs for regularization
    def batch_shuffle(self, dataframe):
        """
            Randomly shuffle pairs of rows in the dataframe into train and validation data
            returns tuple(train_data, valid_data) dataframes
        """
        train_data = pd.DataFrame()
        valid_data = pd.DataFrame()

        # to use the function for other dataframes
        if not dataframe.empty:
            # print('Using custom dataframe')
            self.train_speed_df = dataframe

        for i in range(len(self.train_speed_df) - 1):
            idx1 = np.random.randint(len(self.train_speed_df) - 1)
            idx2 = idx1 + 1
            row1 = self.train_speed_df.iloc[[idx1]].reset_index()
            row2 = self.train_speed_df.iloc[[idx2]].reset_index()

            randInt = np.random.randint(9)
            if 0 <= randInt <= 1:
                valid_frames = [valid_data, row1, row2]
                valid_data = pd.concat(valid_frames, axis=0, join='outer', ignore_index=False)
            if randInt >= 2:
                train_frames = [train_data, row1, row2]
                train_data = pd.concat(train_frames, axis=0, join='outer', ignore_index=False)
        print('Number of train samples:', len(train_data))
        print('Number of validation samples', len(valid_data))
        # print('data type', type(train_data))
        return train_data, valid_data


if __name__ == "__main__":
    pass
    """
    # data directory
    train_data_path = './speed-challenge/data'
    # if Path to raw image folder does not exists make folder
    path_to_images = os.path.join('./speed-challenge/data/IMG')
    if not os.path.exists(path_to_images):
        os.makedirs(path_to_images)
    # reading the train speeds
    train_speed_df = pd.read_csv(os.path.join(
        train_data_path, 'train.txt'), header=None, squeeze=True)
    train_speed_df = pd.DataFrame(
        {'Frame': train_speed_df.index, 'Speed': train_speed_df.values})
    tr_dataset = FrameSpeedDataset(train_data_path, path_to_images, train_speed_df)
    tr_dataset.generate_images_from_video()
    tr_dataset.batch_shuffle()
    """
