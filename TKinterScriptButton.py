# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:25:12 2023

@author: matt8
"""
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import sys
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from Model1 import emotion_dict
import os
from dataset_cleaning import number_of_images, image_dimension
from Model1 import random_sample, gs_to_rgb

df = pd.read_pickle("df_1.pkl")

with open("emo_dict.pkl", "rb") as file:
    emotion_dict = pickle.load(file)

IMDIM = image_dimension(df)
NUMCLASS = df.emotion.nunique()
# Access the boolean values
RGB_or_GS = True
OneHot = True
dir_name = "ModelsRGB{}OneHot{}".format(str(RGB_or_GS), str(OneHot))
dir_name = "TrainedModelsGS"
print(dir_name)
print(emotion_dict)
model_files = os.listdir(dir_name)
loaded_models = {}
for file_name in model_files:
    # Construct the full path to the model file
    model_path = os.path.join(dir_name, file_name)

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Add the loaded model to the dictionary using the file name as the key
    loaded_models[file_name] = model
print(loaded_models)


def get_emo(image_pixels, dict_of_models):
    global emotion_dict, NUMCLASS
    scores = []
    for model in dict_of_models.values():
        scores.append(np.ravel(np.array(model(image_pixels))))
    scores = np.stack(np.array(scores))
    emotions = np.argmax(scores, axis=1)
    unique_values, value_counts = np.unique(emotions, return_counts=True)
    mode_index = np.argmax(value_counts)
    mode = unique_values[mode_index]
    expression = emotion_dict[mode]
    return expression

def start_detection():
    global IMDIM, loaded_models, emotion_dict, cap

    # Create the video capture object
    cap = cv2.VideoCapture(0)

    def process_frame():
        global IMDIM
        # Read frame from the camera and get the matrix of pixel values
        _, frame = cap.read()
        # Convert pixel matrix to grayscale if the model is grayscale-based
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Reduce image size to match the dimensions of the model's reference images
        resized_frame = cv2.resize(gray_frame, (IMDIM, IMDIM))
        # Normalize pixel values to [0,1] interval
        norm_frame = resized_frame / 255
        pixels = norm_frame.reshape(1, IMDIM, IMDIM, 1)
        # Make predictions using the facial model
        emotion = get_emo(pixels, loaded_models)

        # Convert the frame to ImageTk format for displaying in the Tkinter window
        image = Image.fromarray(display_frame)
        image_tk = ImageTk.PhotoImage(image)

        # Update the identified image label
        identified_label.config(image=image_tk)
        identified_label.image = image_tk

        # Update the camera feed label
        video_label.config(image=ImageTk.PhotoImage(Image.fromarray(frame)))
        video_label.image = ImageTk.PhotoImage(Image.fromarray(frame))

        # Update the emotion label
        emotion_label.config(text=emotion)
        window.after(10, process_frame)
    # Start processing the frames
    window.after(0, process_frame)


def start_detection_with_button():
    # Disable the button to prevent multiple detections
    start_button.config(state="disabled")
    # Call the start_detection() function
    start_detection()

# Create the Tkinter window
window = tk.Tk()
window.title("Live Face Recognition")
window.geometry("300x480")

# ... your existing code for labels and other UI elements ...
frame = tk.Frame(window)
frame.pack(side="top")

# Create a label for the identified image
identified_label = tk.Label(frame)
identified_label.pack(side="left", padx=10)

# Create a label for the camera feed
video_label = tk.Label(frame)
video_label.pack(side="right", padx=10)

cap = cv2.VideoCapture(0)

# Create a text label for displaying the emotion
emotion_label = tk.Label(window, text="", font=("Arial", 14))
emotion_label.pack(side="bottom", pady=10)
# Create a button to start the detection
start_button = tk.Button(window, text="Start Detection", command=start_detection_with_button)
start_button.pack(side="top", pady=10)

# Start the Tkinter event loop
window.mainloop()
