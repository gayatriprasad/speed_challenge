# import libraries
import argparse
import os
import csv
import pandas as pd
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Import from local files
from model import nvidia_model
from generator_dataset import generate_training_data, generate_validation_data
from video_to_images_dataset_conversion import FrameSpeedDataset
from predict import predictions, get_pred_mse
from utility_functions import train_valid_split


"""Parse parameters"""
parser = argparse.ArgumentParser(
    description='Detect Speed of the Vehicle from Dashcam Video Challenge')
# hyperparameters settings
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 penalty)')
parser.add_argument('--total_epochs', type=int, default=1, help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, help='pre-trained epochs')
parser.add_argument('--DEVICE_ID', type=int, default=2, help='gpu number')

seed = 999

if __name__ == "__main__":

    global args
    args = parser.parse_args()

    # model
    model = nvidia_model()
    # other cosntants
    model_name = 'nvidia'
    batch_size = 16
    steps_per_epoch = 400

    # datasets
    print("==> Preparing dataset ...")
    data_dir = './speed-challenge/data'
    # if Path to raw image folder does not exists make folder
    path_to_images = os.path.join(data_dir + '/IMG')
    if not os.path.exists(path_to_images):
        os.makedirs(path_to_images)
    # reading the train speeds
    frame_speed_df = pd.read_csv(os.path.join(data_dir, 'train.txt'), header=None, squeeze=True)
    frame_speed_df = pd.DataFrame(
        {'Frame': frame_speed_df.index, 'Speed': frame_speed_df.values})
    fs_dataset = FrameSpeedDataset(data_dir, path_to_images, frame_speed_df)
    # if we want to generate images from input Video (only done once) comment the next line
    # fs_dataset.generate_images_from_video()
    # train_data, valid_data = fs_dataset.batch_shuffle(frame_speed_df)
    # using generator_dataset for faster computation
    # open the generated driving log (image path and speed )
    path_to_processed_csv = os.path.join(data_dir + '/driving_log.csv')
    driving_log_df = pd.read_csv(path_to_processed_csv, header=None)
    print('driving_log_df shape: ', driving_log_df.shape)
    # split the processed data into train and test
    train_data, valid_data = train_valid_split(driving_log_df, seed)

    print('----')
    print('valid_data: ', valid_data.shape)
    print('train_data: ', train_data.shape)

    # get the generated train and generated validation data
    valid_generator = generate_validation_data(valid_data)
    val_size = len(valid_data.index)

    # run the model
    earlyStopping = EarlyStopping(monitor='val_loss', patience=1,
                                  verbose=1, min_delta=0.23, mode='min',)
    #modelCheckpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1, save_weights_only=True)
    #callbacks_list = [modelCheckpoint, earlyStopping]
    callbacks_list = [earlyStopping]
    train_size = len(train_data.index)
    train_generator = generate_training_data(train_data, batch_size)
    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=args.total_epochs,
                                  callbacks=callbacks_list, verbose=1, validation_data=valid_generator, validation_steps=val_size)

    # printing some metrics
    print('history:', history)
    print()
    print('Ended Training')


    # calculate the validation score
    val_score = model.evaluate_generator(test_generator, steps=len(test))
    print()
    print('validation score:', val_score)
    data = predictions(test, model)
    data.to_pickle('speed-challenge/data/predictions.pkl')
    print(get_pred_mse('/speed-challenge/data/predictions.pkl'))
