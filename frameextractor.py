# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021
This python file for extracting key frames from video files was given a part of the Project_Part2_SourceCode zip file.
It was changed to fit the project's needs. It extracts key frames at 65% (Originally 50%) of the video length and saves them as PNG files.
OpenAI's ChatGPT (version GPT-4) was used to clean up and optimize the file.

Attribution:
- OpenAI. "ChatGPT Language Model." Version GPT-4. Accessed October 2, 2024. https://chat.openai.com/.

@author: chakati
"""

# Code to extract a key frame from a video and save it as a PNG file.

import os
import cv2

def frameExtractor(videopath, frames_path, count):
    """
    Extracts a key frame (at 65% of video length) from the given video file
    and saves it as a PNG image in the specified directory.

    :param videopath: Path to the video file.
    :param frames_path: Directory where the extracted frames will be saved.
    :param count: Counter to assign an order number to the extracted frame.
    :return: The path to the saved frame, or None if extraction fails.
    """
    try:
        # Ensure the frames directory exists
        os.makedirs(frames_path, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(videopath)

        # Get the total number of frames in the video
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        # Ensure the video has frames
        if video_length <= 0:
            print(f"Error: Video {videopath} has no frames.")
            return None

        # Calculate the frame number to extract (65% of the video length)
        frame_no = int(video_length * 0.65)
        print(f"Extracting frame {frame_no} from video {videopath} (Total frames: {video_length})")

        # Set the video position to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Could not read frame {frame_no} from video {videopath}")
            return None

        # Create the filename for saving the frame
        filename = os.path.join(frames_path, f"{count + 1:05d}.png")

        # Save the extracted frame as a PNG file
        cv2.imwrite(filename, frame)
        print(f"Frame saved to {filename}")

        return filename

    except Exception as e:
        print(f"Error in frame extraction: {str(e)}")
        return None

    finally:
        # Release the video capture object
        cap.release()