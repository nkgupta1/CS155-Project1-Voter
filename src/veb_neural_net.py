#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from veb_common import *
from sklearn.neural_network import MLPClassifier

X_train, Y_train = normalized_categorized_data()


def NNC(layers, tol=0.0001, max_iter=100000, cv=True):
    '''
    param: cv: if cv is false, does not cross validate, trains on full data, 
    and pickles the result.
    param: frac_train: fraction of training to use for training (if False, 
    use all of the training data)
    '''
    print('learning', layers, tol, max_iter, cv)
    start = time.time()
    X, Y = X_train, Y_train
    clf = MLPClassifier(hidden_layer_sizes=layers, verbose=True, tol=tol, max_iter=max_iter)
    if cv:
        scores = cross_val_score(clf, X, Y, cv=4, 
            scoring='accuracy', verbose=True)
        print('cross-validation accuracy:', np.mean(scores))
    else:
        clf.fit(X, Y)
        finish = time.time()
        score = clf.score(X_train, Y_train)
        print score
        # print np.mean(clf.predict(X_train)), np.mean(Y_train)
        save_name = 'd-MLP-' + str(layers) + '-' + str(max_iter)
        save_name += '-' + str(score) + '-' + str(int(time.time() - start))
        joblib.dump(clf, 'saved_models/' + save_name)
    print('finished in', (time.time() - start), 'seconds')



NNC((500, 150), tol=.00001, max_iter=2, cv=True)

#clf_to_prediction(2008, 'd-MLP-200-1-0.79777939289-47')


