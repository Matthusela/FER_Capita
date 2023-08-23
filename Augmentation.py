# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:49:37 2023

@author: matt8
"""
import os
# Would need to be changed on another machine.
if os.getcwd() != 'C:\\Users\\matt8\\fer_capita':
    os.chdir('C:\\Users\\matt8\\fer_capita')
import pandas as pd
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from dataset_cleaning import pixels_mat,emotions

# Rather than reading from csv like in dataset_cleaning file, create df from 
# already processed variables for speed and clarity. 
emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
df = pd.DataFrame(index = range(len(emotions)),data = {"pixels":pixels_mat,"emotion":emotions})
df["expression"] = df.emotion.map(emotion_dict)

# Class count stats
emo_counts = df.emotion.value_counts()
emo_counts_arr = np.array(df.emotion.value_counts())
emo_std = emo_counts_arr.std()
emo_mean = emo_counts_arr.mean()
classInstances = math.floor(emo_mean)

# Bar chart to show the initial class imbalance present in the data, specifically,
# the lack of data for the disgust emotion
bar_chart1 = plt.bar(emo_counts.index, emo_counts.values)
bar_width = round([bar.get_width() for bar in bar_chart1][0]/2,2)
plt.hlines(classInstances,-bar_width,len(emo_counts_arr)-1+bar_width,colors="red",alpha = 0.7)
plt.title("initial Class DIstribution")
plt.show()

# Could be generalised to account for any combination of emotions but not necessary
MakeDisgustAnger = True
if MakeDisgustAnger == True:
    df_emo1_index = df[df.emotion == 1].copy().index
    df.loc[df_emo1_index,"emotion"] = 0
    df.loc[df_emo1_index,"expression"] = emotion_dict[0]
    
    new_emotions,new_ems_num = df.expression.unique(),df.expression.nunique()
    emo_keys = list(range(new_ems_num))
    new_dict = {}
    for emo in range(len(new_emotions)):
        new_dict[new_emotions[emo]] = emo_keys[emo]
    emotion_dict = {value: key for key, value in new_dict.items()}
    df["emotion"] = df.expression.map(new_dict)
    
    
    emo_counts = df.emotion.value_counts()
    emo_counts_arr = np.array(df.emotion.value_counts())
    emo_std = emo_counts_arr.std()
    emo_mean = emo_counts_arr.mean()
    classInstances = math.floor(emo_mean)
    
    bar_chart_mix = plt.bar(emo_counts.index, emo_counts.values)
    bar_width = round([bar.get_width() for bar in bar_chart_mix][0]/2,2)
    plt.hlines(classInstances,-bar_width,len(emo_counts_arr)-1+bar_width,colors="red",alpha = 0.7)
    plt.title("Class DIstribution post disgust + anger combination")
    plt.show()

# Randomly samples the classInstances number of images from each emotion which
# is over represented in the initial dataset.
def sample_heavy(emotion):
    global classInstances
    temp = df[df.emotion == emotion].copy()
    kept = temp.sample(classInstances)
    return kept

# Defining so that the .map call in the augment_class function is neater
def flip(matrix):
    return cv2.flip(matrix,1)

# Applies the image flip for the underrepresented classes - checks if there are
# enough images in that class that flipping at least each image once is sufficient
# to reach classInstance value. Adds flipped images to temp class dataframe.
def augment_class(emotion):
    global classInstances
    temp = df[df.emotion == emotion].copy()
    disparity = classInstances - len(temp)
    if disparity < len(temp):
        print("{} images being flipped - so not every image of emotion {} is augmented".format(disparity,emotion_dict[emotion]))
        to_augment = temp.sample(disparity)

    else: 
        print("Not enough instances of {} emotion, each image flipped once but class still under mean!".format(emotion_dict[emotion]))
        to_augment = temp.sample(frac=1)
    to_augment["pixels"] = to_augment.pixels.map(flip)
    temp = pd.concat([temp,to_augment])
    return temp

# Combines the treatments for the over and underrepresented classes 
augments = {}
truth = dict(emo_counts > classInstances)
for i in truth.keys():
    name = "emotion_" + str(i)
    if truth[i]:
        # overrepresented classes
        augments[name] = sample_heavy(i)
    else:
        #underrepresented classes
        augments[name] = augment_class(i)
        
# Combines all individual classes into one df and then randomises by sampling whole df
aug_df = pd.concat(augments.values()).sample(frac=1)
aug_df = aug_df.drop("expression",axis=1)
# Recalculating some stats
aug_emo_counts = aug_df.emotion.value_counts()
aug_emo_counts_arr = np.array(aug_emo_counts)
aug_emo_mean = aug_emo_counts_arr.mean()
classInstances = math.floor(aug_emo_mean)
# Formatting new class balanced/augmented dataframe
aug_df.columns = ["pixels_mat","emotion"]
aug_df.index = range(len(aug_df.index))

# Final bar chart, all classes should be balanced
bar_chart_aug = plt.bar(aug_emo_counts.index, aug_emo_counts.values)
bar_width = round([bar.get_width() for bar in bar_chart_aug][0]/2,2)
plt.hlines(classInstances,-bar_width,len(aug_emo_counts_arr)-1+bar_width,colors="red",alpha = 0.7)
plt.title("final augmented Class DIstribution")
plt.show()
print("Augmentation Complete")



