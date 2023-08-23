# Initial external Module import
import os
if os.getcwd() != 'C:\\Users\\matt8\\fer_capita':
    os.chdir('C:\\Users\\matt8\\fer_capita')
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import cv2

# Import project file functions and variables
# Image dimension will not change so inp_dim from dataset_cleaning does not need
# to be recalculated 
from dataset_cleaning import inp_dim
from df_functions import number_of_images,image_dimension

# If augmented classes are to be used then want to use the dataframe produced
# from the augmentation file, otherwise still merge the anger and disgust emotions
# and use dataframe from dataset_cleaning sheet.
AUGMENT = True
if AUGMENT == True:
    from Augmentation import aug_df,emotion_dict
    df = aug_df.copy()
# If augmented images are not used then this merging emotions process may throw up errors if the script is run multiple times without kernel being restarted,
# sometimes tries to depopulate dictionary again which leads to errors. This is all to deal with the merging of the anger and disgust emotion which is also
# covered in Augmentation file. In general AUGMENT should be True though so should not throw errors
else:
    merge_emotions = True
    from dataset_cleaning import df,emotion_dict 
    if merge_emotions == True:
        if len(emotion_dict) == 7:
          df.loc[df.emotion == 1,"emotion"] = 0
          # Reflects changes from having merged emotions.
          df["emotion"] = df.emotion.map({0:0,2:1,3:2,4:3,5:4,6:5})
          new_emotion = {}
          emotion_dict.pop(1)
          for key,value in enumerate(emotion_dict):
              new_emotion[key] = emotion_dict[value]
          emotion_dict = new_emotion
      
# How many classes are now in data - if anger and disgust merged should be 6, otherwise 7
NumberClass = len(emotion_dict)

print("Classes are balanced True or False: {}".format(
    (np.array(df.emotion.value_counts())==np.array(df.emotion.value_counts())[0]).all()))

# Equivalent of SKLearn split,test train. Seed construction is weird and probably
# unnecessary
def random_sample(data_frame,random,train_perc = 0.8):
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

# Using cv2 this function takes a list of greyscale images and converts them to 
# RGB images, necessary for some models and adds rgb images to dataframe
def gs_to_rgb(list_of_gs_imgs, output = False):
  stacks = []
  for img in np.stack(list_of_gs_imgs):
      stacks.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
  stacks = np.stack(np.array(stacks))
  df["preprocessed_images"] = list(stacks)
  if output == True:
    return stacks
 
# Retrieves X_train, X_test, Y_train, Y_test matrices depending on the RGB and
# one-hot specifications, order everything needs to be done in is quite precise, is
# set up in such a way that when models are fitting they are each fitted on a 
# different train and test set.  
def get_sets(df,one_hot,rgb):
    global inp_dim
    if rgb == True:
      gs_to_rgb(np.stack(df.pixels_mat.values), output = False)
    train_df,test_df,train_labels,test_labels = random_sample(df,True,0.9)
    train_size = number_of_images(train_df)
    test_size = number_of_images(test_df)
    Y_train = train_labels.values.reshape(-1,1)
    Y_test = test_labels.values.reshape(-1,1)
    if one_hot == True:
        test_labels = np.squeeze(tf.one_hot(Y_test,NumberClass))
        Y_test = np.squeeze(test_labels)
        train_labels = np.squeeze(tf.one_hot(Y_train,NumberClass))
        Y_train = np.squeeze(train_labels)
    if rgb == True:
      X_train = np.stack(train_df.preprocessed_images.values)/255
      X_test = np.stack(test_df.preprocessed_images.values)/255
    else:
      X_train = np.stack(train_df.pixels_mat.values).reshape(train_size,inp_dim,inp_dim,1)/255
      X_test = np.stack(test_df.pixels_mat.values).reshape(test_size,inp_dim,inp_dim,1)/255
    shape = X_train.shape
    return X_train,Y_train,X_test,Y_test, shape


