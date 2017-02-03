#!/usr/bin/env python3

"""Commonly used functions, just so that we don't have to rewrite them
in every file.
"""

import numpy as np
import csv

def importTrain(filename):
    """
    Imports training data and parses it
    """
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]
    data_array = np.asarray(data)
    # remove first row (column labels), first column (id) and last column
    # (label)
    X_train = data_array[1:,1:-1]
    # remove first row (column labels), keep last column (label)
    Y_train = data_array[1:,-1]
    return X_train, Y_train

def importTest(filename):
    """
    Imports test data and parses it
    """
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]
    data_array = np.asarray(data)
    # remove first row (column labels), first column (id)
    X_train = data_array[1:,1:]
    return X_train
    
