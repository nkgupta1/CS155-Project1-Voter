#!/usr/bin/env python3
"""
A naive test on the 2008 data.
"""

from common import importTrain
import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

X_train_2008, Y_train_2008 = importTrain('../data/train_2008.csv')


