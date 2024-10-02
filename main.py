# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

This Python file for gesture training and testing using cosine similarity was primarily developed by the project developer.
It handles the extraction of key frames from videos, feature extraction using a CNN model, and gesture recognition.
OpenAI's ChatGPT (version GPT-4) was used to guide the developer when stuck and to clean up and optimize the file.

Attribution:
- OpenAI. "ChatGPT Language Model." Version GPT-4. Accessed October 2, 2024. https://chat.openai.com/.

@author: chakati
"""
import csv
import os
from importlib.metadata import files

import cv2
import tensorflow as tf

# Custom Imports
from frameextractor import frameExtractor
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
    """
    Extracts features from a frame in the input video file.

    :param file_folder: The directory containing video files.
    :param input_file: The video file to extract the frame from.
    :param frame_folder: The directory to save the extracted frame.
    :param counter: A counter for frame saving.
    :return: Extracted handshape features from the frame or None if failed.
    """
    try:
        # Extract the key frame from the video
        print(f"Extracting frame from {input_file}")
        frame_path = frameExtractor(os.path.join(file_folder, input_file), frame_folder, counter)

        if frame_path is None:
            print(f"Failed to extract frame from {input_file}")
            return None

        # Read the extracted frame as a grayscale image
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image {frame_path}")
            return None

        # Extract handshape features from the image
        print(f"Extracting features from frame {frame_path}")
        extractor = HandShapeFeatureExtractor.get_instance()
        features = extractor.extract_feature(image)

        if features is None:
            print(f"Feature extraction failed for image {frame_path}")
            return None

        print(f"Features extracted successfully from {frame_path}")
        return features

    except Exception as e:
        print(f"Error in get_feature: {str(e)}")
        return None


# Mapping gesture names to corresponding numeric labels
train_dict = {
    "Num0": 0, "Num1": 1, "Num2": 2, "Num3": 3, "Num4": 4, "Num5": 5,
    "Num6": 6, "Num7": 7, "Num8": 8, "Num9": 9, "FanDown": 10, "FanOff": 11,
    "FanOn": 12, "FanUp": 13, "LightOff": 14, "LightOn": 15, "SetThermo": 16
}

test_dict = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "DecreaseFanSpeed": 10, "FanOff": 11, "FanOn": 12,
    "IncreaseFanSpeed": 13, "LightOff": 14, "LightOn": 15, "SetThermo": 16
}

# =============================================================================
# Process training data
# =============================================================================
train_data_list = []
train_data_folder = "./train/"
train_frames_folder = "./train_frames"
count = 0

# Ensure there are 51 PNG files in the train_frames folder
train_frames = [file for file in os.listdir(train_frames_folder) if file.endswith(".png") and not file.startswith('.')]
train_frames = train_frames[:51]
assert len(train_frames) == 51, f"Expected 51 PNG images in train_frames folder but found {len(train_frames)}"

# Process only .mp4 files
train_files = [file for file in os.listdir(train_data_folder) if file.endswith(".mp4") and not file.startswith('.')]
train_files = train_files[:51]
assert len(train_files) == 51, f"Expected 51 train files but found {len(train_files)}"

for train_file in train_files:
    print(f"Processing training video: {train_file}")

    # Extract features from the training video
    features = get_feature(train_data_folder, train_file, train_frames_folder, count)

    if features is None:
        print(f"Feature extraction failed for {train_file}")
        continue

     # Safely split the filename and handle different lengths
    x = train_file.split('_')

    # Extract gesture name (first part)
    gest_name = x[0]

    # Extract gesture version (third part, if available)
    gest_version = x[2] if len(x) > 2 else 'Unknown'

    # Lookup gesture number in the dictionary
    gest_num = train_dict.get(gest_name, -1)  # Assign -1 if the gesture is not found in the dictionary

    # Store the training data
    train_info = TrainInfo(gest_name, gest_version, gest_num, features)
    train_data_list.append(train_info)

    count += 1

# =============================================================================
# Process test data and match with training data using cosine similarity
# =============================================================================
test_folder = "./test/"
test_frame_folder = "./test_frames"
count = 0
out_data = []

# Ensure there are 51 PNG files in the test_frames folder
test_frames = [file for file in os.listdir(test_frame_folder) if file.endswith(".png") and not file.startswith('.')]
test_frames = test_frames[:51]
assert len(test_frames) == 51, f"Expected 51 PNG images in test_frames folder but found {len(test_frames)}"

# Process only valid .mp4 files from the test folder
test_files = [file for file in os.listdir(test_folder) if file.endswith(".mp4") and not file.startswith('.')]
test_files = test_files[:51]
assert len(test_files) == 51, f"Expected 51 test files but found {len(test_files)}"

for test_file in test_files:
    print(f"Processing test video: {test_file}")

    # Extract features from the test video
    test_features = get_feature(test_folder, test_file, test_frame_folder, count)

    if test_features is None:
        print(f"Feature extraction failed for {test_file}")
        continue

    # Parse gesture info from filename
    test_gest_name = test_file.split('-')[2]
    test_gest_num = test_dict.get(test_gest_name)

    # Compare the test features with all training features
    match = None
    min_cos_sim = float('inf')

    for train in train_data_list:
        cosine_similarity = tf.keras.losses.cosine_similarity(test_features, train.gest_feature, axis=-1)
        cos_sim = float(cosine_similarity.numpy())

        if cos_sim < min_cos_sim:
            min_cos_sim = cos_sim
            match = train

    out_data.append([match.gest_num])
    print(f"Test gesture {test_gest_num} matched with train gesture {match.gest_num}")
    count += 1

# Write results to CSV file
with open("./Results.csv", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(out_data)
    print("Results saved to Results.csv")
