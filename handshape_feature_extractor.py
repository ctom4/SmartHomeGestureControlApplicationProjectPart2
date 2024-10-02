"""
This Python file for the HandShapeFeatureExtractor singleton class was given a part of the Project_Part2_SourceCode zip file.
It was changed to fit the project's needs. It loads and holds a CNN model in memory and handles the feature extraction from video frames.
OpenAI's ChatGPT (version GPT-4) was used to clean up and optimize the file.

Attribution:
- OpenAI. "ChatGPT Language Model." Version GPT-4. Accessed October 2, 2024. https://chat.openai.com/.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model

"""
This is a Singleton class which bears the ML model in memory.
The model is used to extract handshape features.
"""

BASE = os.path.dirname(os.path.abspath(__file__))


class HandShapeFeatureExtractor:
    __single = None

    @staticmethod
    def get_instance():
        """
        Returns the singleton instance of the HandShapeFeatureExtractor.
        If no instance exists, it creates one.
        """
        if HandShapeFeatureExtractor.__single is None:
            HandShapeFeatureExtractor()
        return HandShapeFeatureExtractor.__single

    def __init__(self):
        if HandShapeFeatureExtractor.__single is None:
            try:
                # Load the CNN model from the specified path
                model_path = os.path.join(BASE, 'cnn_model.h5')
                self.model = load_model(model_path)
                print("CNN model loaded successfully.")

                # Set singleton instance
                HandShapeFeatureExtractor.__single = self
            except Exception as e:
                print(f"Error loading CNN model: {str(e)}")
                raise
        else:
            raise Exception("This Class bears the model, so it is made Singleton")

    @staticmethod
    def __pre_process_input_image(crop):
        """
        Preprocesses the image for model input:
        - Resizes to 200x200 pixels.
        - Normalizes the pixel values to the range [0, 1].
        - Reshapes it for the CNN input.

        :param crop: The cropped image to preprocess.
        :return: Preprocessed image ready for the model.
        """
        try:
            # Resize and normalize the image
            img = cv2.resize(crop, (200, 200))
            img_arr = np.array(img) / 255.0
            img_arr = img_arr.reshape(1, 200, 200, 1)
            return img_arr
        except Exception as e:
            print(f"Error during image preprocessing: {str(e)}")
            raise

    @staticmethod
    def __bound_box(x, y, max_y, max_x):
        """
        Calculate bounding box dimensions for cropping specific hand parts.

        :param x: X-coordinate of the center.
        :param y: Y-coordinate of the center.
        :param max_y: Maximum Y boundary.
        :param max_x: Maximum X boundary.
        :return: Bounding box coordinates for cropping.
        """
        try:
            y1 = min(y + 80, max_y)
            y2 = max(y - 80, 0)
            x1 = min(x + 80, max_x)
            x2 = max(x - 80, 0)
            return y1, y2, x1, x2
        except Exception as e:
            print(f"Error during bounding box calculation: {str(e)}")
            raise

    def extract_feature(self, image):
        """
        Extracts features from a preprocessed image using the loaded CNN model.

        :param image: The input image to extract features from.
        :return: Extracted features.
        """
        try:
            # Preprocess the input image
            img_arr = self.__pre_process_input_image(image)
            print(f"Extracting features from processed image of shape {img_arr.shape}")

            # Predict features using the model
            features = self.model.predict(img_arr)
            print(f"Features extracted successfully: {features}")
            return features
        except Exception as e:
            print(f"Error during feature extraction: {str(e)}")
            raise

