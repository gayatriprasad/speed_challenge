# import libraries
import pickle
import numpy as np
import pandas as pd
import model
import os
from tensorflow.keras.models import load_model
from video_to_images_dataset_conversion import FrameSpeedDataset


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


def predictions(data, model):
    new_data = []
    for idx in range(1, len(data.index) - 1):
        row_now = data.iloc[[idx]].reset_index()
        row_prev = data.iloc[[idx - 1]].reset_index()
        row_next = data.iloc[[idx + 1]].reset_index()

        time_now = float(row_now[1].values[0])
        time_prev = float(row_prev[1].values[0])
        time_next = float(row_next[1].values[0])

        if abs(time_now - time_prev) == 1 and time_now > time_prev:
            row1 = row_prev
            row2 = row_now

        elif abs(time_next - time_now) == 1 and time_next > time_now:
            row1 = row_now
            row2 = row_next
        else:
            print('Error generating row')

        x1, y1 = fsd.preprocess_image_from_path(row1[0].values[0], row1[2].values[0])
        x2, y2 = fsd.preprocess_image_from_path(row2[0].values[0], row2[2].values[0])

        img_diff = fsd.optical_flow(x1, x2)
        img_diff = img_diff.reshape(1, img_diff.shape[0], img_diff.shape[1], img_diff.shape[2])
        y = np.mean([y1, y2])

        prediction = model.predict(img_diff)
        error = abs(prediction - y2)
        new_data.append([prediction[0][0], y2, error[0][0], time_now, time_prev])
    return pd.DataFrame(new_data)


def get_pred_mse(preds):
    df = pd.read_pickle(preds)
    avg = np.mean(df[2].values**2)
    return avg


if __name__ == "__main__":
    pass
    """
    model = model.speed_model()
    model.load_weights("./model-weights.h5")
    df = pd.read_csv("./processed.csv", header=None)
    pre = PreProcessor()
    train, test = fsd.shuffle_frame_pairs(df)
    test_generator = training.generate_validation_data(test)
    val_score = model.evaluate_generator(test_generator, steps=len(test))
    data = predictions(test, model)
    data.to_pickle("./predictions.pkl")
    print(get_pred_mse("predictions.pkl"))
    """
