#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from copy import deepcopy as copy


X_train, Y_train = train_2008_categorized()


def NNC(layers, tol=0.0001, max_iter=100000, cv=True, early_stopping=False, 
    validation_fraction=0.1):
    '''
    param: cv: if cv is false, does not cross validate, trains on full data, 
    and pickles the result.
    param: frac_train: fraction of training to use for training (if False, 
    use all of the training data)
    '''
    print('learning', layers, tol, max_iter, cv)
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes=layers, verbose=True, tol=tol, 
        max_iter=max_iter, early_stopping=early_stopping, 
        validation_fraction=validation_fraction)
    if cv:
        scores = cross_val_score(clf, X_train, Y_train, cv=5, scoring='accuracy', verbose=True)
        score = np.mean(scores)
        print('cross-validation accuracy:', score)
    else:
        clf = VotingClassifier([('first', clf), ('second', clf), 
            ('third', clf), ('fourth', clf), ('fifth', clf), 
            ('sixth', clf), ('seventh', clf), ('eigth', clf), 
            ('ninth', clf), ('tenth', clf)], voting='soft', n_jobs=1)
        clf.fit(X_train, Y_train)
        score = clf.score(X_train, Y_train)
        #print np.mean(clf.predict(X_train))
        print('score:', score)
        save_name = 'd-MLP_10ensemble-' + str(layers) + '-' + str(max_iter)
        save_name += '-' + str(score) + '-' + str(int(time.time() - start))
        joblib.dump(clf, 'saved_models/' + save_name)
    print('finished in', (time.time() - start), 'seconds')


NNC((150, 50, 10), tol=.00001, max_iter=9, cv=True, early_stopping=False)

#clf_to_prediction('d-MLP_10ensemble-(100, 50, 10)-8-0.807676249091-1108', 2008)

#analyze_ensemble('d-MLP_10ensemble-(100, 50, 10)-8-0.807676249091-1108')
