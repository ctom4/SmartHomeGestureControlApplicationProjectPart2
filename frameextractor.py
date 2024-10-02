# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
# code to get the key frame from the video and save it as a png file.

import cv2
import os


# videopath: path of the video file
# frames_path: path of the directory to which the frames are saved
# count: to assign the video order to the frame.
def frameExtractor(videopath, frames_path, count):
    # Ensure frames path exists
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)

    cap = cv2.VideoCapture(videopath)

    # Get the total number of frames in the video
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Ensure the video has frames
    if video_length <= 0:
        print(f"Error: Video {videopath} has no frames.")
        return None

    # Calculate which frame to extract (65% of the video length)
    frame_no = int(video_length * 0.65)
    print(f"Extracting frame {frame_no} from video {videopath} (Total frames: {video_length})")

    # Set the video to the specific frame
    cap.set(1, frame_no)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame {frame_no} from video {videopath}")
        return None

    # Save the extracted frame
    filename = frames_path + "/%#05d.png" % (count + 1)
    cv2.imwrite(filename, frame)
    print(f"Frame saved to {filename}")

    return filename