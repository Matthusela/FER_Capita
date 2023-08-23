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
if os.getcwd() != 'C:\\Users\\matt8\\fer_capita':
    os.chdir('C:\\Users\\matt8\\fer_capita')
import sys
import cv2
from tensorflow.keras.layers import  Conv2D
from tensorflow.keras.layers import  MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import  Dense
from tensorflow.keras import datasets
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers

from Model1 import NumberClass,get_sets,df

# Instantiates the CNN model architecture, the InputShape argument is only specified
# later on when this function is actually called. The architecture is probably not
# the best that can be found - however grid searching is too intensive for my laptop
# and this is why have introduced the randomise argument to try and find a decent
# architecture for the model - could probably also do with having much higher freedom 
# in randomness
def gen_model(randomise,InputShape):
    global NumberClass
    # Initial layer is same for every model
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
    # final layer is same for every model     
    model.add(Dense(NumberClass, activation="softmax"))    
    return model

# Defines the learning rate, sets the optimiser and loss functions for the models
# haven't introduced much variability here.
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

# Takes the architecture and loss and optimisation functions and begins to fit the 
# model - this is the step where if everything has worked out properly the code should
# happily run when this function is actually called. epochs determines how many attempts
# are made to fit the model, while stopping is used to cut fitting process off early
# if no progress appears to be being made. This process is time and computing intensive.
def fit_model(train_data,train_targets,test_data,test_targets, epoch_iterations,model,*stopping):
    if stopping:
        model = model.fit(x = train_data,y= train_targets,epochs= epoch_iterations,
                         validation_data=(test_data, test_targets),callbacks= stopping)
    else: 
        model = model.fit(x = train_data,y= train_targets,epochs= epoch_iterations,
                         validation_data=(test_data, test_targets))
    return model 


# Combines the three previous functions into one function and keeps some stats
# on each model
def Process(train_data,train_targets,
            test_data,test_targets,learning_rate,
            epoch_iterations,Inp_shape,OneHot,randomised):
    instantiate = gen_model(randomised,Inp_shape)
    cnn = config_model(instantiate, learning_rate,OneHot)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    cnn_history = fit_model(train_data, train_targets, test_data, test_targets, epoch_iterations, cnn,early_stopping)
    loss, accuracy = cnn.evaluate(test_data, test_targets)
    return cnn,cnn_history,accuracy

# Big process of generating the models by calling above Process() function multiple times
# and ensuring models get saved in specific folders based on their RGB and One-Hot
# properties
def Model_Files(one_hot_labels,rbg_or_greyscale):
    global df
    # All this path and directory code is just to save models in specific folders based on RGB and
    # one-hot settings.
    dir_name = "ModelsRGB{}OneHot{}".format(str(rbg_or_greyscale),str(one_hot_labels))
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Specify the relative path to the new folder
    relative_path = dir_name
    # Create the full path to the new folder
    save_directory = os.path.join(current_directory, relative_path)
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if os.path.exists(save_directory):
        models_already = len(os.listdir(save_directory))
    # Print statements just to check everytjing is as expected    
    print("working directory is {}".format(os.getcwd()))
    print("rgb? {}".format(rbg_or_greyscale))
    print("OneHot? {}".format(one_hot_labels))
    print("Save directory is {}".format(save_directory))
    # Now process of calling the models at different learning rates begins:
    if models_already:
        count = models_already
    else:
        count = 0
    print(count)

    # For each of the three learning rates here 5 different models are fitted - this can easily be changed and I am sure some learning
    # rates are better than others though I am unsure which. The models are only saved to files if they have an accuracy exceeding 
    # 50%
    
    for i in range(5):
        learn = 0.001
        X_train,Y_train,X_test,Y_test,InputShape = get_sets(df,one_hot_labels,rbg_or_greyscale)
        model = Process(train_data = X_train,train_targets =Y_train,
                       test_data = X_test,
                       test_targets = Y_test,learning_rate = learn,
                       epoch_iterations = 15,Inp_shape = InputShape,
                       OneHot = one_hot_labels,randomised = True)
        name = "ModelNo{}".format(count+1)
        if model[2] > 0.5:
            count += 1
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
        name = "ModelNo{}".format(count+1)
        if model[2] > 0.5:
            count += 1
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
        name = "ModelNo{}".format(count+1)
        if model[2] > 0.5:
            count += 1
            save_path = os.path.join(save_directory, f"{name}.h5")
            tf.keras.models.save_model(model[0], save_path)
            print("model {} saved to {}".format(name,dir_name))
    
    
  
    
