#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.externals import joblib

def normalized_data():
    X_train, Y_train = train_2008()
    #for arr in [X_train, Y_train]:
    #   print arr.shape, arr.dtype
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    std[std == 0] = 1  # in case values are constant for entire column
    X_train = (X_train - mean) / std
    return X_train, Y_train
    
def save_prediction(name, clf, year=2008):
    X_train, Y_train = train_2008()
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    std[std == 0] = 1
    print('saving', name)
    if year == 2008:
        X_test, id_test = test_2008()
    elif year == 2012:
        X_test, id_test = test_2012()
    X_test = (X_test - mean) / std
    result = format_results(id_test, clf.predict(X_test))
    with open('submissions/' + str(year) + '-' + name + '.csv', 'w') as f:
        f.write(result)

def clf_to_prediction(year, name):
    save_prediction(name, joblib.load('saved_models/' + name), year)

#clf_to_prediction(2008, 'd-MLP-(100, 50, 10)-0.02-0.790294895387-6')


