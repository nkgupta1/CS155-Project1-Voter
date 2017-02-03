#!/usr/bin/env python3
"""
A naive test on the 2008 data.
"""

import numpy as np 
# import tensorflow as tf 
# import keras
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation, Flatten, Dropout

X_train_2008 = np.genfromtxt('../data/test_2008.csv', delimiter=",")
print(X_train_2008.shape)
Y_train_2008 = X_train_2008[:,-1]
X_train_2008 = X_train_2008[:,:-1]
print(X_train_2008.shape)
print(Y_train_2008.shape)
# print(X_train_2008)
