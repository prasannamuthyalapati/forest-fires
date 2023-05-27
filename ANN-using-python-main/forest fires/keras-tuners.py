# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QCuRWVmLigXx8JSoWtVAS7E0qyS1UQtd
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
#to select no of layers,nodes,learning rate for best model we use keras tuners (generates hyper parameters)
from kerastuner.tuners import RandomSearch

from google.colab import files
upload=files.upload()

# load pima indians dataset

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

dataset

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

def build_model(hp):
  model=keras.Sequential()
  for i in range(hp.Int('num_layers',2,20)):
    model.add(layers.Dense(units=hp.Int('units_'+str(i),min_value=32,max_value=512,step=32),activation='relu'))
    model.add(layers.Dense(1,activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',[1e-2,1e-3,1e-4])),loss='mean_absolute_error',metrics=['mean_absolute_error'])
  return model

tuner=RandomSearch(build_model,objective='val_mean_absolute_error',max_trials=5,executions_per_trial=3)

tuner.search_space_summary()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

tuner.search(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

tuner.results_summary()

#here score is mean absolute error, so which ever have least score we consider those hyperparameters