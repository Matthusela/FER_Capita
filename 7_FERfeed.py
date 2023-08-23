# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:25:12 2023

@author: matt8
"""
# External Modules
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import os
if os.getcwd() != 'C:\\Users\\matt8\\fer_capita':
    os.chdir('C:\\Users\\matt8\\fer_capita')
import tensorflow as tf
import numpy as np

# Project files
from Model1 import emotion_dict
from dataset_cleaning import inp_dim

IMDIM = inp_dim
NUMCLASS = len(emotion_dict)
# Access the boolean values - these 2 lines as talked about in the ReadMe file are currently unnecessary and were introduced to 
# easily retrieve specific models in conjuction with directory naming convention from Model_Files() of RandomModels file.
# Do not think that code below is set up for models trained on RGB rather than GreyScale images.
RGB_or_GS = False
OneHot = True
# The dir_name variable reflects where the trained expression models have been saved - likely to be necessary to change this line
dir_name = "AugmentModels"
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
    
print(list(loaded_models.keys()))

# This is the CV2 algorithm that has been introduced to identify faces in images (see 23/08/23 note in ReadMe)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


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
        global IMDIM, face_cascade
        # Read frame from the camera and get the matrix of pixel values
        _, frame = cap.read()
        
        # Convert pixel matrix to grayscale if the model is grayscale-based
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # The faces uses a pre-trained facial detection model to identify a face
        # in the image. The face area is then padded out slightly in order to match
        # the characteristic padding of the training images
        faces = face_cascade.detectMultiScale(display_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            pad = 40 
            cv2.rectangle(display_frame, (x-pad, y-pad),
                          (x + w+pad, y + h+pad), (255, 0, 0), 2)
        
        # If the face_cascade model has detected a face, then use the bounded region,
        # shown by the rectangle in the live feed, and perform the emotion detection
        # on just this region.
        if len(faces) > 0:
            image_frame = gray_frame[y-pad:y+h+pad, x-pad:x+w+pad]
            calibrated_image_frame = cv2.resize(image_frame, (IMDIM, IMDIM)) / 255.0
            pixels = calibrated_image_frame.reshape(1, IMDIM, IMDIM, 1)
            emotion = get_emo(pixels, loaded_models)
            
        else:
            emotion = "No face present"

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
window.title("Live Expression Recognition")
window.geometry("200x200")

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
