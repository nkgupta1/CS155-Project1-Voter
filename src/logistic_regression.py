#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from veb_common import *

threads = 2

X_train, Y_train = normalized_data()

in_sample_accuracy = lambda clf: np.mean(clf.predict(X_train) == Y_train)


def logistic_regression(max_iter=100, C=1., cv=True, frac_train=False):
    '''
    param: cv: if cv is false, does not cross validate, trains on full data, 
    and pickles the result.
    param: frac_train: fraction of training to use for training (if False, 
    use all of the training data)
    '''
    print('learning', max_iter, C, cv, frac_train)
    start = time.time()
    if frac_train:
        X = X_train[:int(frac_train * X_train.shape[0])]
        Y = Y_train[:int(frac_train * Y_train.shape[0])]
    else:  X, Y = X_train, Y_train
    clf = LogisticRegression(verbose=True, n_jobs=threads, solver='liblinear',
        max_iter=max_iter, C=C)
    if cv:
        scores = cross_val_score(clf, X, Y, cv=5, 
            scoring='accuracy', verbose=True)
        print('cross-validation accuracy:', np.mean(scores))
    else:
        clf.fit(X, Y)
        finish = time.time()
        # print np.mean(clf.predict(X_train)), np.mean(Y_train)
        save_name = 'b-LOGR_liblinear-' + str(max_iter) + '-' + str(C) + '-'
        save_name += str(in_sample_accuracy(clf)) + '-'
        save_name += str(frac_train) + '-' + str(int(time.time() - start))
        joblib.dump(clf, 'saved_models/' + save_name)
    print('finished in', (time.time() - start), 'seconds')



logistic_regression(max_iter=100, C=1, cv=True)

#clf_to_prediction(2008, 'a-SVM_linear-0.762351740455-0.1-124')


