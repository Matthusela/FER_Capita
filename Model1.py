# Initial Module import
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys
import cv2
from tensorflow.keras.layers import  Conv2D
from tensorflow.keras.layers import  MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import  Dense
from tensorflow.keras import datasets
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers

# Bringing dataframe in from dataset_cleaning.py

from dataset_cleaning import number_of_images,image_dimension,emotion_dict,df


print("df produced")


###############################  Defining some functions and steps before training can begin ###############################

# Equivalent of SKLearn split,test train
def random_sample(data_frame,random,train_perc = 0.8,test_perc=0.2):
    ind = np.array(data_frame.index)
    entries = len(ind)
    rand_Seed = np.random.randint(0,entries)
    fixed_Seed = entries
    train_num = round(entries*train_perc)
    if random == True:  
        rng = np.random.default_rng(rand_Seed)
        shuff = rng.choice(ind,size = entries,replace = False)
        train_sample = data_frame.iloc[shuff][:train_num]
        train_labels = data_frame.iloc[shuff].emotion[:train_num]
        test_sample = data_frame.iloc[shuff][train_num:]
        test_labels = data_frame.iloc[shuff].emotion[train_num:]
    elif random == False:
        rng = np.random.default_rng(fixed_Seed)
        shuff = rng.choice(ind,size = entries,replace = False)
        train_sample = data_frame.iloc[shuff][:train_num]
        test_sample = data_frame.iloc[shuff][train_num:]
        train_labels = data_frame.iloc[shuff].emotion[:train_num]
        test_labels = data_frame.iloc[shuff].emotion[train_num:]
    return train_sample.drop("emotion",axis = 1),test_sample.drop("emotion",axis = 1),train_labels,test_labels

# Using cv2 this function takes a list of greyscale images and converts them to RGB images, necessary for some models and adds rgb images to dataframe
def gs_to_rgb(list_of_gs_imgs, output = False):
  stacks = []
  for img in np.stack(list_of_gs_imgs):
      stacks.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
  stacks = np.stack(np.array(stacks))
  df["preprocessed_images"] = list(stacks)
  if output == True:
    return stacks
    
NumberClass = len(emotion_dict)



def get_sets(df,one_hot,rgb):
    if rgb == "rgb":
      gs_to_rgb(np.stack(df.pixels_mat.values), output = False)
    train_df,test_df,train_labels,test_labels = random_sample(df,True,0.9,True)
    train_size = number_of_images(train_df)
    test_size = number_of_images(test_df)
    pixel_dim = image_dimension(df)
    Y_train = train_labels.values.reshape(-1,1)
    Y_test = test_labels.values.reshape(-1,1)
    if one_hot == True:
        test_labels = np.squeeze(tf.one_hot(Y_test,NumberClass))
        Y_test = np.squeeze(test_labels)
        train_labels = np.squeeze(tf.one_hot(Y_train,NumberClass))
        Y_train = np.squeeze(train_labels)
    if rgb == "rgb":
      #gs_to_rgb(np.stack(df.pixels_mat.values), output = False)
      X_train = np.stack(train_df.preprocessed_images.values)/255
      X_test = np.stack(test_df.preprocessed_images.values)/255
    else:
      X_train = np.stack(train_df.pixels_mat.values).reshape(train_size,pixel_dim,pixel_dim,1)/255
      X_test = np.stack(test_df.pixels_mat.values).reshape(test_size,pixel_dim,pixel_dim,1)/255
    shape = X_train.shape
    return X_train,Y_train,X_test,Y_test, shape
############################################             initiating model set up           ###########################################



   
  ############################################             instantiating the model           ###########################################


