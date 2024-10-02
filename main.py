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
    def __init__(self, gest_name ,gest_vers, gest_num, gest_feature):
        self.gest_name = gest_name
        self.gest_vers = gest_vers
        self.gest_num = gest_num
        self.gest_feature = gest_feature

def get_feature(file_folder, input_file, frame_folder, counter):

    fname1 = frameExtractor(file_folder + input_file, frame_folder, counter)
    image = cv2.imread(fname1, cv2.IMREAD_GRAYSCALE)
    extract_feat = HandShapeFeatureExtractor.extract_feature(HandShapeFeatureExtractor.get_instance(), image)
    return extract_feat


train_dict= {
                "Num0" : 0,
                "Num1" : 1,
                "Num2" : 2,
                "Num3" : 3,
                "Num4" : 4,
                "Num5" : 5,
                "Num6" : 6,
                "Num7" : 7,
                "Num8" : 8,
                "Num9" : 9,
                "FanDown" : 10,
                "FanOff" : 11,
                "FanOn" : 12,
                "FanUp" : 13,
                "LightOff" : 14,
                "LightOn" : 15,
                "SetThermo" : 16
                }
test_dict= {
                "0" : 0,
                "1" : 1,
                "2" : 2,
                "3" : 3,
                "4" : 4,
                "5" : 5,
                "6" : 6,
                "7" : 7,
                "8" : 8,
                "9" : 9,
                "DecreaseFanSpeed" : 10,
                "FanOff" : 11,
                "FanOn" : 12,
                "IncreaseFanSpeed" : 13,
                "LightOff" : 14,
                "LightOn" : 15,
                "SetThermo" : 16
                }


# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

train_data_list = []
train_data_folder = "./train/"
train_frames_folder = "./train_frames"
count = 0


for train_file in os.listdir(train_data_folder):
    print(train_file)
    x = train_file.split('_')

    gest_name = x[0]
    gest_version = x[2]
    gest_num = train_dict.get(x[0])

    features = get_feature(train_data_folder, train_file, train_frames_folder, count)

    train_info = TrainInfo(gest_name, gest_version, gest_num, features)
    train_data_list.append(train_info)

    #print("train_info {} {} {}".format(train_info.gest_name, train_info.gest_vers, train_info.gest_num))
    count += 1


# =============================================================================
# Get the penultimate layer for test data and recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

test_folder = "./test/"
test_frame_folder = "./test_frames"
count = 0
out_data = []

for test_file in os.listdir(test_folder):

    test_file2 = test_file.replace('.','-')
    x = test_file2.split('-')
    test_gest_vers = x[0]
    test_gest_name = x[2]
    test_gest_num = test_dict.get(x[2])

    test_features = get_feature(test_folder, test_file, test_frame_folder, count)

    print(test_file)
    print("test_data_info {}, {}, {}, {}".format(test_file, test_gest_vers, test_gest_name, test_gest_num))

    match = train_data_list[0]
    min_test_val = 1000
    for train in train_data_list:
        cosine_similarity = tf.keras.losses.cosine_similarity(test_features, train.gest_feature, axis=-1)
        cos_sim = float(cosine_similarity.numpy())
        print("\t\t --> test_data {} == train_data {}, {}, {} cos_sim = {}  min_test_val = {} ".format(test_gest_num,
                                                train.gest_num, train.gest_name, train.gest_vers, cos_sim, min_test_val))

        if cos_sim < min_test_val:
            min_test_val = cos_sim
            gest_num = train.gest_name
            match = train
            print(" \t\t --->  new min = {} new gest_num = {}".format(min_test_val, match.gest_num))

    #out_data.append([test_gest_name, match.gest_num])
    out_data.append([match.gest_num])
    print("xxxxxx---------------> test_data_gest_num = {}  train_data_gest_num = {}".format(test_gest_num, match.gest_num ))
    count +=1

with open("./Results.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(out_data)