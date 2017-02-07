#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from veb_common import *
from sklearn.neural_network import MLPClassifier

X_train, Y_train = normalized_data()

in_sample_accuracy = lambda clf: np.mean(clf.predict(X_train) == Y_train)

def SVM(layers, tol=0.0001, cv=True):
    '''
    param: cv: if cv is false, does not cross validate, trains on full data, 
    and pickles the result.
    param: frac_train: fraction of training to use for training (if False, 
    use all of the training data)
    '''
    print('learning', layers, tol, cv)
    start = time.time()
    X, Y = X_train, Y_train
    clf = MLPClassifier(hidden_layer_sizes=layers, verbose=True, tol=tol)
    if cv:
        scores = cross_val_score(clf, X, Y, cv=3, 
            scoring='accuracy', verbose=True)
        print('cross-validation accuracy:', np.mean(scores))
    else:
        clf.fit(X, Y)
        finish = time.time()
        score = clf.score(X_train, Y_train)
        print score
        # print np.mean(clf.predict(X_train)), np.mean(Y_train)
        save_name = 'd-MLP-' + str(layers) + '-' + str(tol)
        save_name += '-' + str(score) + '-' + str(int(time.time() - start))
        joblib.dump(clf, 'saved_models/' + save_name)
    print('finished in', (time.time() - start), 'seconds')



SVM((100, 50, 10), tol=.02, cv=False)

#clf_to_prediction(2008, 'a-SVM_linear-0.762351740455-0.1-124')


