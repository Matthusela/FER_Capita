# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 23:34:20 2023

@author: matt8
"""

# Ensemble Prep 
# Initial Module import
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os
import sys
import cv2
from tensorflow.keras.layers import  Conv2D
from tensorflow.keras.layers import  MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import  Dense
from tensorflow.keras import datasets
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers

from Model1 import NumberClass,get_sets
df = pd.read_pickle("df_1.pkl")
def gen_model(randomise,InputShape):
    global NumberClass 
    model = tf.keras.Sequential()
    model.add(Conv2D(64,(3,3),activation = "relu",input_shape = InputShape[1:]))
    if randomise == False:
        model.add(layers.BatchNormalization())
        model.add(AveragePooling2D(2,2))
        model.add(Conv2D(64,(4,4),activation = "relu"))
        model.add(MaxPooling2D(3,3))
        model.add(Conv2D(128,(8,8),activation = "relu"))
        model.add(MaxPooling2D(3,3))
        model.add(layers.Flatten())       
        model.add(layers.BatchNormalization())
        model.add(Dense(128,activation="relu",kernel_initializer="he_normal"))
        model.add(Dense(64,activation="relu", kernel_initializer="he_normal"))
        model.add(Dense(32,activation="relu", kernel_initializer="he_normal"))
        model.add(layers.Dropout(0.5))
        
    elif randomise == True:
        if np.random.randint(2) == 1:
            model.add(layers.BatchNormalization())
        if np.random.randint(2) == 1:
            model.add(AveragePooling2D(2,2))
            model.add(Conv2D(64,(3,3),activation = "relu"))
        if np.random.randint(2) == 1:
            model.add(MaxPooling2D(3,3))
            model.add(Conv2D(128,(4,4),activation = "relu"))
        if np.random.randint(2) == 1:
            model.add(MaxPooling2D(3,3))
        model.add(layers.Flatten())              
        if np.random.randint(2) == 1:
            model.add(layers.BatchNormalization())
        if np.random.randint(4) == 1:
            model.add(Dense(128,activation="relu"))
        if np.random.randint(2) == 1:
            model.add(Dense(64,activation="relu"))
        if np.random.randint(2) == 1:
            model.add(Dense(32,activation="relu", kernel_initializer="he_normal"))
        if np.random.randint(2) == 1:
            model.add(Dense(16,activation="relu", kernel_initializer="he_normal"))
        if np.random.randint(2) == 1:
            model.add(layers.Dropout(0.5)) 
    else:
        print("True or False")
        
    model.add(Dense(NumberClass, activation="softmax"))    
    
    return model
 
def config_model(model,learningRate,one_hot_labels):
    if one_hot_labels == True:
          model.compile(optimizer= optimizers.Adam(learning_rate=learningRate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    else:
      model.compile(optimizer= optimizers.Adam(learning_rate=learningRate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    return model


def fit_model(train_data,train_targets,test_data,test_targets, epoch_iterations,model,*stopping):
    if stopping:
        model = model.fit(x = train_data,y= train_targets,epochs= epoch_iterations,
                         validation_data=(test_data, test_targets),callbacks= stopping)
    else: 
        model = model.fit(x = train_data,y= train_targets,epochs= epoch_iterations,
                         validation_data=(test_data, test_targets))
    return model 



def Process(train_data,train_targets,
            test_data,test_targets,learning_rate,
            epoch_iterations,Inp_shape,OneHot,randomised):
    instantiate = gen_model(randomised,Inp_shape)
    cnn = config_model(instantiate, learning_rate,OneHot)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    cnn_history = fit_model(train_data, train_targets, test_data, test_targets, epoch_iterations, cnn,early_stopping)
    loss, accuracy = cnn.evaluate(test_data, test_targets)
    return cnn,cnn_history,accuracy

def Model_Files(one_hot_labels,rbg_or_greyscale):
    global df
    dir_name = "ModelsRGB{}OneHot{}".format(str(rbg_or_greyscale),str(one_hot_labels))
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Specify the relative path to the new folder
    relative_path = dir_name
    # Create the full path to the new folder
    save_directory = os.path.join(current_directory, relative_path)
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    #os.chdir(dir_name)
    print("working directory is {}".format(os.getcwd()))
    print("rgb? {}".format(rbg_or_greyscale))
    print("OneHot? {}".format(one_hot_labels))
    count = []
    for i in range(5):
        learn = 0.001
        X_train,Y_train,X_test,Y_test,InputShape = get_sets(df,one_hot_labels,rbg_or_greyscale)
        model = Process(train_data = X_train,train_targets =Y_train,
                       test_data = X_test,
                       test_targets = Y_test,learning_rate = learn,
                       epoch_iterations = 15,Inp_shape = InputShape,
                       OneHot = one_hot_labels,randomised = True)
        name = "ModelNo{}".format(len(count)+1)
        if model[2] > 0.3:
            count.append(1)
            save_path = os.path.join(save_directory, f"{name}.h5")
            tf.keras.models.save_model(model[0], save_path)
            print("model {} saved to {}".format(name,dir_name))
        
    for i in range(5):
        learn = 0.0001
        X_train,Y_train,X_test,Y_test,InputShape = get_sets(df,one_hot_labels,rbg_or_greyscale)
        model = Process(train_data = X_train,train_targets =Y_train,
                       test_data = X_test,test_targets = Y_test,
                       learning_rate = learn , epoch_iterations = 15,
                       Inp_shape = InputShape,OneHot = one_hot_labels,randomised = True)
        name = "ModelNo{}".format(len(count)+1)
        if model[2] > 0.3:
            count.append(1)
            save_path = os.path.join(save_directory, f"{name}.h5")
            tf.keras.models.save_model(model[0], save_path)
            print("model {} saved to {}".format(name,dir_name))
            
    for i in range(5):
        learn = 0.01
        X_train,Y_train,X_test,Y_test,InputShape = get_sets(df,one_hot_labels,rbg_or_greyscale)
        model = Process(train_data = X_train,train_targets =Y_train,
                       test_data = X_test,test_targets = Y_test,
                       learning_rate = learn ,
                       Inp_shape = InputShape,OneHot = one_hot_labels,
                       epoch_iterations = 15,randomised = True)
        name = "ModelNo{}".format(len(count)+1)
        if model[2] > 0.3:
            count.append(1)
            save_path = os.path.join(save_directory, f"{name}.h5")
            tf.keras.models.save_model(model[0], save_path)
            print("model {} saved to {}".format(name,dir_name))
    
  
    
