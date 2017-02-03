#!/usr/bin/env python3

import numpy as np
import time
import csv

def importTrain(filename):
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]
    data_array = np.asarray(data)
    # remove first row, first column and last column
    X_train = data_array[1:,1:-1]
    # remove first row, keep last column
    Y_train = data_array[1:,-1]
    return X_train, Y_train

def importTest(filename):
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]
    data_array = np.asarray(data)
    # remove first row, first column
    X_train = data_array[1:,1:]
    return X_train
    
