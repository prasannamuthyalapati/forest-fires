# -*- coding: utf-8 -*-
"""neural_nw.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MnhzvvI6VmIU9neG9icw9RfCQpa_f0B_
"""

import numpy as np
import pandas as pd



from google.colab import files

upload=files.upload()

# load pima indians dataset

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

dataset

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

#train test splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#initialization ANN
model=Sequential()

#adding input and 1st hidden layer
model.add(Dense(units=12,kernel_initializer='he_uniform', input_dim=8, activation='relu'))
#adding 2nd hidden layer
model.add(Dense(units=8,kernel_initializer='he_uniform', activation='relu'))
#adding output layer
model.add(Dense(units=1,kernel_initializer='glorot_uniform', activation='sigmoid'))



# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, validation_split=0.33, batch_size=10,epochs=100)

y_pred=model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,Y_test)


# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))