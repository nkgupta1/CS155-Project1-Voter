#!/usr/bin/env python3

"""Commonly used functions, just so that we don't have to rewrite them
in every file.
"""

import numpy as np
import csv

def mode(ndarray,axis=0):
    """
    Calculates the by-dimension mode, i.e. majority votes, of a np array.

    Code courtesy of devdev on stackoverflow, copied and modified from:
    http://stackoverflow.com/questions/16330831/most-efficient-way-to-find-
        mode-in-numpy-array
    """

    if ndarray.size == 1:
        return (ndarray[0],1)
    elif ndarray.size == 0:
        raise Exception('Attempted to find mode on an empty array!')

    try:
        axis = [i for i in range(ndarray.ndim)][axis]
    except IndexError:
        raise Exception('Axis %i out of range for array with %i dimension(s)' % 
            (axis,ndarray.ndim))

    srt = np.sort(ndarray,axis=axis)
    dif = np.diff(srt,axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1,-1) 
        for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices,axis=axis)
    location = np.argmax(bins,axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() 
        for i in range(bins.ndim)]
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    return modals

def import_train(filename):
    """
    Imports training data and parses it.
    Removes column labels and id column
    Converts entries to ints
    Converts labels to 0 to not vote and 1 to vote
    """
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]

    # remove first row (column labels), first column (id)
    data_array = np.asarray(data)[1:, 1:]

    # convert to 32-bit int
    data_array = data_array.astype(np.int32)
     
    # remove last column (label)
    X_train = data_array[:, :-1]
    # keep last column (label) and make into one hot vector
    # remove first entry in one hot vector because doesn't correspond to anything
    Y_train = 2 - data_array[:, -1]
    

    return X_train, Y_train

def import_test(filename):
    """
    Imports test data and parses it
    Removes column labels and id column
    Converts entries to ints
    """
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]
    # remove first row (column labels), first column (id)
    data_array = np.asarray(data)[1:, 1:]

    # convert to 32-bit int
    X_train = data_array.astype(np.int32)

    return X_train
    
def predictions_to_number(y_labels):
    """
    Receives input of list of did not vote (0) or vote (1) required format of
    not vote (2) or vote (1)
    """
    return 2 - y_labels

