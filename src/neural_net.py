#!/usr/bin/env python3
"""
A naive test on the 2008 data.
"""

from common import *
import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout

def net(X, Y):
    model = Sequential()

    # model.add(Dense(750, input_dim=381))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))

    model.add(Dense(1, input_dim=381))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.5))

    # model.add(Dense(25))
    # model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='mse',optimizer='adagrad', metrics=['accuracy'])

    fit = model.fit(X, Y, batch_size=128, nb_epoch=5, verbose=1)

    return model

def save(model, filename):
    model.save('saved_models/' + filename + '-arch.h5')
    model.save_weights('saved_models/' + filename + '-weights.h5')

def load(filename):
    model = load_model('saved_models/' + filename + '-arch.h5')
    model.load_weights('saved_models/' + filename + '-weights.h5')
    return model

def eval(model, X, Y):
    score = model.evaluate(X, Y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    

X_train_2008, Y_train_2008 = train_2008()
model = net(X_train_2008, Y_train_2008)
save(model, 'nn-500-250')
 
# X_test, ids = test_2008() 
# model = load('nn-500-250')
# print(model.get_weights())
# predictions = model.predict(X_test)
# print(sum(predictions))

# to try and prevent an exception
import gc; gc.collect()