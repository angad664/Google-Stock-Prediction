#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:41:23 2019

@author: angadsingh
"""

# business problem - we gonna predict the google stock price (open column) for year 2017

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#feature Scaling
from sklearn.preprocessing import MinMaxScaler #we gonna use normalization which is (x- min(x))/(max(x)-min(x))
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# create a data structure with 60 timestep and 1 output
X_train = []
y_train = []
for i in range(60,1258): # we start from 60 because it need 60 timestep to predict 1 output
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping - we want to add dimension in our np array
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#part 2
# building RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialize the RNN
regressor = Sequential()

#adding first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # true bc we have to add another lstm layer
regressor.add(Dropout(0.2)) # 20 percent will be ignored during propogation and backpropagation

#adding second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

##adding fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

# adding output layer
regressor.add(Dense(units = 1)) # stock price at time t+1 which is what we have to predict

#compile the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting rnn to training_set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# part 3 making prediction and visualizing the results

# importing real stock prices of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# getting predicted stock price of 2017
# we have to scale again in order to concat so, we have to take the original dataset and test dataset
# to concat . we are only scaling the input not the output. our rnn is trained on scaled input means
# we have to normalize

dataset_total = pd.concat(dataset_train['Open'], dataset_test['Open'], axis = 0) # axis = 0 is vertical??
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80): # 80 because we only have 20days to predict
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # to inverse the scaling of prediction. in order to get original scaling

#visualize the result
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend
plt.show()
