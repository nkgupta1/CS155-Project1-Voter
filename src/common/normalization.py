#!/usr/bin/env python3

"""Commonly used functions, just so that we don't have to rewrite them
in every file.
"""

import numpy as np
from sklearn.externals import joblib
from categorize import *

def normalized_data():
    X_train, Y_train = train_2008()
    #for arr in [X_train, Y_train]:
    #   print arr.shape, arr.dtype
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    std[std == 0] = 1  # in case values are constant for entire column
    X_train = (X_train - mean) / std
    return X_train, Y_train

def normalized_categorized_data():
    X_train, Y_train = train_2008_categorized()
    X_train, Y_train = X_train, Y_train
    #for arr in [X_train, Y_train]:
    #   print arr.shape, arr.dtype
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    std[std == 0] = 1  # in case values are constant for entire column
    X_train = (X_train - mean) / std
    return X_train, Y_train

