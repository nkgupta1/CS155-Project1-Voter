#!/usr/bin/env python3

"""Commonly used functions, just so that we don't have to rewrite them
in every file.
"""

import numpy as np
import csv
from sklearn.externals import joblib

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
    # keep last column (label) and map 2 -> 0 (not vote) and 1 -> 1 (vote)
    Y_train = 2 - data_array[:, -1]
    
    return X_train, Y_train

def import_test(filename):
    """
    Imports test data and parses it
    Removes column labels and id column
    Converts entries to ints
    Return ids separately
    """
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]

    # remove first row (column labels)
    data_array = np.asarray(data)[1:, :]

    # keep only first column (id)
    ids = data_array[:, 0]

    # remove first column (id)
    data_array = np.asarray(data_array)[:, 1:]


    # convert to 32-bit int
    X_train = data_array.astype(np.int32)

    return X_train, ids
    
def save_parsed_data():
    """
    Saves data as binary in parsed format
    Much quicker to run
    Must be run from src
    """
    X_train_2008, Y_train_2008 = import_train('../data/raw/train_2008.csv')
    test_2008, id_2008 = import_test('../data/raw/test_2008.csv')
    test_2012, id_2012 = import_test('../data/raw/test_2012.csv')

    joblib.dump(X_train_2008, '../data/X_train_2008.pkl')
    joblib.dump(Y_train_2008, '../data/Y_train_2008.pkl')

    joblib.dump(test_2008, '../data/X_test_2008.pkl')
    joblib.dump(id_2008, '../data/id_test_2008.pkl')

    joblib.dump(test_2012, '../data/X_test_2012.pkl')
    joblib.dump(id_2012, '../data/id_test_2012.pkl')

def train_2008():
    """
    Gets data and labels from the 2008 training set
    Must be run from src
    """
    return joblib.load('../data/X_train_2008.pkl'), joblib.load('../data/Y_train_2008.pkl')

def test_2008():
    """
    Gets data from the 2008 test set
    Must be run from src
    """
    return joblib.load('../data/X_test_2008.pkl'), joblib.load('../data/id_test_2008.pkl')

def test_2012():
    """
    Gets data from the 2012 test set
    Must be run from src
    """
    return joblib.load('../data/X_test_2012.pkl'), joblib.load('../data/id_test_2012.pkl')

def predictions_to_number(y_labels):
    """
    Receives input of list of did not vote (0) or vote (1) required format of
    not vote (2) or vote (1)
    """
    return 2 - y_labels

def format_results(ids, Y):
    """
    Takes in a 1d array of ids and a 1d array of predictions where 0 is not vote
    and 1 is vote
    """
    Y = predictions_to_number(Y)
    to_ret = 'id,PES1\n'
    for i in range(len(ids)):
        to_ret += ids[i] + ',' + str(Y[i]) + '\n'
    return to_ret



