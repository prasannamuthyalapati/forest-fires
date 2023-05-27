# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:38:18 2022

@author: chaitanya
"""
!pip install tensorflow-gpu
import pandas as pd 
import numpy as np
data=pd.read_csv(r"C:\Users\chaitanya\OneDrive\Desktop\naresh it\New folder\cls note nit\ANN-using-python-main\ANN-using-python-main\forestfires.csv")
df=data.drop(data.columns[[0,1,2]],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['size_category']=le.fit_transform(df['size_category'])

x=df.iloc[:,:-1]
y=df.iloc[:,-1]



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from  keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential() 
model.add(Dense(units=10,activation='relu',kernel_initializer='he_uniform',input_dim=27))
#adding 2nd hidden layer
model.add(Dense(units=8,activation='relu',kernel_initializer='he_uniform'))
#adding output layer
model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))


model.compile(loss = 'mean_squared_error',  optimizer = 'sgd', metrics = ['accuracy'])
print(model.summary())
model.fit(x_train, y_train, batch_size = 32, epochs = 250,validation_data = (x_test, y_test),validation_split=0.33)
Y_predict = model.predict(x_test)
print(Y_predict)
Y_predict = (Y_predict > 0.5)
print(Y_predict)
accuracy = model.evaluate(x_test, y_test)

print(accuracy*100,"accurancy:%.2f")
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
history = model.fit(x_train, y_train, epochs=1)
keras_clf = KerasClassifier(model.summary)
accuracy = cross_val_score(estimator=keras_clf, scoring="accuracy",X=x_train, y=y_train, cv=5)
print(accuracy)
