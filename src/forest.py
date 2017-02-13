#!/usr/bin/env python3

from common import *
import numpy as np
from sklearn.model_selection import KFold

def accuracy(y, z):
    '''
    For two sets of labels, calculates accuracy
    '''

    assert len(y) == len(z)

    hits = 0
    for i in range(len(y)):
        if y[i] == z[i]:
            hits += 1
    hits /= len(y)

    return hits

def refine_predict(Y_predict):
    '''
    Given raw predictions, sets top 75% to 1, else 0
    '''

    Y_range = Y_predict[:]
    Y_range = sorted(Y_range)
    div = Y_range[int(round(0.75*len(Y_predict)-1))]

    for i in range(len(Y_predict)):
        if Y_predict[i] >= div:
            Y_predict[i] = 1
        else:
            Y_predict[i] = 0
    return Y_predict

if __name__=='__main__':
    # Caveat; need classes to be zero-indexed
    X, Y = train_2008_categorized()
    print('Dataset Loaded')
    kf = KFold(n_splits=5)

    for d in [1]:#,2,3]:
        print('Train to depth {0}'.format(str(d)))
        for n_est in [50,200,500]:
            print('  N_est: {0}'.format(str(n_est)))
            count = 1
            for train_index, test_index in kf.split(X):
                print('    Using validation set {0}'.format(str(count)))
                X_train = X[train_index]
                Y_train = Y[train_index].astype(int)

                X_test = X[test_index]
                Y_test = Y[test_index].astype(int)

                f = forest('saved_models/test_forest',model='extra')
                print('      Training...')
                f.fit(X_train, Y_train, depth=d,n_estimators=n_est)

                print('      Predicting...')
                Y_predict = f.predict(X_test).astype(int)

                # Y_predict = refine_predict(Y_predict)

                print('      Comparing predictions...')
                acc = accuracy(Y_test, Y_predict)
                print('      Accuracy: {0}'.format(str(acc)))

                count += 1
                quit()