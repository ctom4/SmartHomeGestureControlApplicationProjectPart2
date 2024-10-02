# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
from frameextractor import frameExtractor
import csv

## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor


# =============================================================================
# Classes and Data
# =============================================================================
class TrainInfo:
    def __init__(self, gest_name, gest_vers, gest_num, gest_feature):
        self.gest_name = gest_name
        self.gest_vers = gest_vers
        self.gest_num = gest_num
        self.gest_feature = gest_feature


def get_feature(file_folder, input_file, frame_folder, counter):
    try:
        # Extract the key frame from the video
        print(f"Extracting frame from {input_file}")
        fname1 = frameExtractor(file_folder + input_file, frame_folder, counter)

        if fname1 is None:
            print(f"Failed to extract frame from {input_file}")
            return None

        # Read the image from the extracted frame
        image = cv2.imread(fname1, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image {fname1}")
            return None

        # Extract handshape features from the image
        print(f"Extracting features from frame {fname1}")
        extract_feat = HandShapeFeatureExtractor.get_instance().extract_feature(image)

        if extract_feat is None:
            print(f"Feature extraction failed for image {fname1}")
            return None

        print(f"Features extracted successfully from {fname1}")
        return extract_feat

    except Exception as e:
        print(f"Error in get_feature: {str(e)}")
        return None


train_dict = {
    "Num0": 0,
    "Num1": 1,
    "Num2": 2,
    "Num3": 3,
    "Num4": 4,
    "Num5": 5,
    "Num6": 6,
    "Num7": 7,
    "Num8": 8,
    "Num9": 9,
    "FanDown": 10,
    "FanOff": 11,
    "FanOn": 12,
    "FanUp": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16
}

test_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "DecreaseFanSpeed": 10,
    "FanOff": 11,
    "FanOn": 12,
    "IncreaseFanSpeed": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16
}

# =============================================================================
# Get the penultimate layer for training data
# =============================================================================

train_data_list = []
train_data_folder = "./train/"
train_frames_folder = "./train_frames"
count = 0

# Ensure that the train_frames folder contains exactly 51 PNG files
train_frames = [file for file in os.listdir(train_frames_folder) if file.endswith(".png") and not file.startswith('.')]
assert len(train_frames) == 51, f"Expected 51 PNG images in train_frames folder but found {len(train_frames)}"

# Process only .mp4 files and skip hidden files
train_files = [file for file in os.listdir(train_data_folder) if file.endswith(".mp4") and not file.startswith('.')]

for train_file in train_files:
    print(f"Processing training video: {train_file}")

    # Extract features from the training video
    features = get_feature(train_data_folder, train_file, train_frames_folder, count)

    if features is None:
        print(f"Feature extraction failed for {train_file}")
        continue

    x = train_file.split('_')
    gest_name = x[0]
    gest_version = x[2]
    gest_num = train_dict.get(x[0])

    train_info = TrainInfo(gest_name, gest_version, gest_num, features)
    train_data_list.append(train_info)

    count += 1

# =============================================================================
# Get the penultimate layer for test data and recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

test_folder = "./test/"
test_frame_folder = "./test_frames"
count = 0
out_data = []

# Ensure that the test_frames folder contains exactly 51 PNG files
test_frames = [file for file in os.listdir(test_frame_folder) if file.endswith(".png") and not file.startswith('.')]
assert len(test_frames) == 51, f"Expected 51 PNG images in test_frames folder but found {len(test_frames)}"

# Ensure that the frames folder contains exactly 51 PNG files
frames_folder = "./frames/"
frames = [file for file in os.listdir(frames_folder) if file.endswith(".png") and not file.startswith('.')]
assert len(frames) == 51, f"Expected 51 PNG images in frames folder but found {len(frames)}"

# Filter valid .mp4 files in the test folder and skip hidden files
test_files = [file for file in os.listdir(test_folder) if file.endswith(".mp4") and not file.startswith('.')]

# Ensure there are exactly 51 test files
assert len(test_files) == 51, f"Expected 51 test files but found {len(test_files)}"

for test_file in test_files:
    print(f"Processing test video: {test_file}")

    # Extract features from the test video
    test_features = get_feature(test_folder, test_file, test_frame_folder, count)

    if test_features is None:
        print(f"Feature extraction failed for {test_file}")
        continue

    test_file2 = test_file.replace('.', '-')
    x = test_file2.split('-')
    test_gest_vers = x[0]
    test_gest_name = x[2]
    test_gest_num = test_dict.get(x[2])

    match = train_data_list[0]
    min_test_val = 1000
    for train in train_data_list:
        cosine_similarity = tf.keras.losses.cosine_similarity(test_features, train.gest_feature, axis=-1)
        cos_sim = float(cosine_similarity.numpy())
        print(
            f"\t--> Test data {test_gest_num} vs. Train data {train.gest_num}, {train.gest_name}, {train.gest_vers} cos_sim = {cos_sim}")

        if cos_sim < min_test_val:
            min_test_val = cos_sim
            match = train
            print(f"\t\t--> New minimum: {min_test_val}, Gesture: {match.gest_num}")

    out_data.append([match.gest_num])
    print(f"Test gesture {test_gest_num} matched with train gesture {match.gest_num}")
    count += 1

# Ensure only 51 results are written
assert len(out_data) == 51, f"Expected 51 results but found {len(out_data)}"

# Write results to CSV file with UTF-8 encoding
with open("./Results.csv", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(out_data)
    print("Results saved to Results.csv")