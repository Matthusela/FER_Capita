import pandas as pd
import numpy as np
import math
from df_functions import image_dimension, number_of_images
# Read kaggle data set from wherever on your computer saved
df = pd.read_csv(r"C:\Users\matt8\Downloads\faceset1\icml_face_data.csv")

# Ignore usage column
df = df.drop(" Usage",axis=1).copy()

# From source above this is how the emotion codes correlate to the emotions
emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

# Converting pixel data into usable form that TF (tensorflow) will like later on 
df.columns = ["emotion","pixels"]
df["pixels_list"] = df.pixels.map(lambda x: np.fromstring(x, dtype=np.uint16, sep=' '))

# Functions now defined in df_functions as less clunky having them in separate file
inp_dim = image_dimension(df)
num_dim = number_of_images(df)

# Image representation is by standard a sqaure grid of pixels, from this dataset each image is a 2D square matrix of size 48
# Reflecting the matrix is done through:
pixels_mat =  df.pixels_list.apply(lambda x : x.reshape(inp_dim,inp_dim))
df["pixels_mat"] = pixels_mat
# Unnecessary info so unnecessary memory usage.
df = df.drop(["pixels","pixels_list"],axis=1)
# Necessary for Augmentation sheet
emotions = df.emotion


