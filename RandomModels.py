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
import sys
import cv2
from tensorflow.keras.layers import  Conv2D
from tensorflow.keras.layers import  MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import  Dense
from tensorflow.keras import datasets
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers

from Model1 import InputShape,NumberClass,X_test,X_train,Y_train,Y_test,one_hot_labels

def gen_model(randomise):
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
 
def config_model(model,learningRate):
    global one_hot_labels
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
            test_data,test_targets,learning_rate, epoch_iterations, randomised):
    instantiate = gen_model(randomised)
    cnn = config_model(instantiate, learning_rate)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    cnn_history = fit_model(train_data, train_targets, test_data, test_targets, epoch_iterations, cnn,early_stopping)
    loss, accuracy = cnn.evaluate(test_data, test_targets)
    return cnn,cnn_history,accuracy

#Mod1 = Process(train_data = X_train,train_targets =Y_train,
       #        test_data = X_test,test_targets = Y_test,
     #m           learning_rate = 0.01, epoch_iterations = 5,randomised = True)

#new_arch = Process(train_data = X_train,train_targets =Y_train,
               #test_data = X_test,test_targets = Y_test,
            #  learning_rate = 0.01, epoch_iterations = 5,randomised = False)

ModelDic = {}
for i in range(15):
    learn = 0.001
    model = Process(train_data = X_train,train_targets =Y_train,
                   test_data = X_test,test_targets = Y_test,
                   learning_rate = learn , epoch_iterations = 15,randomised = True)
    name = "ModelNo{}rate{}".format(i,learn)
    ModelDic[name] = {"model" : model[0], "history" : model[1], "accuracy":model[2]}
    
for i in range(15):
    learn = 0.0001
    model = Process(train_data = X_train,train_targets =Y_train,
                   test_data = X_test,test_targets = Y_test,
                   learning_rate = learn , epoch_iterations = 15,randomised = True)
    name = "ModelNo{}rate{}".format(i,learn)
    ModelDic[name] = model

for i in range(20):
    learn = 0.01
    model = Process(train_data = X_train,train_targets =Y_train,
                   test_data = X_test,test_targets = Y_test,
                   learning_rate = learn , epoch_iterations = 15,randomised = True)
    name = "ModelNo{}rate{}".format(i,learn)
    ModelDic[name] = model

