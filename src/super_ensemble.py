#!/usr/bin/env python3

from common import *
import numpy as np
import time
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def Ensemble_NN_RF(clfs):
    print('starting ensemble of', len(clfs), 'clfs')  # for mental pacification
    X_train, Y_train = train_2008_categorized()
    start = time.time()
    clf = VotingClassifier([(str(i), clfs[i]) for i in range(len(clfs))], 
            voting='soft')
    clf.fit(X_train, Y_train)
    score = clf.score(X_train, Y_train)
    print('score:', score)
    save_name = 'e-ENN1-' + str(score) + '-' + str(int(time.time() - start))
    joblib.dump(clf, 'saved_models/' + save_name)
    print('finished in', (time.time() - start), 'seconds')
    return save_name



if __name__=='__main__':

    # clf_rf = RandomForestClassifier(n_estimators=1000)
    clf_nn_A = MLPClassifier(hidden_layer_sizes=(100,50,10), verbose=False, 
        tol=0.00001, max_iter=9, early_stopping=False)
    clf_nn_B = MLPClassifier(hidden_layer_sizes=(150,50,10), verbose=False, 
        tol=0.00001, max_iter=6, early_stopping=False)
    clf_nn_C = MLPClassifier(hidden_layer_sizes=(150,50), verbose=False, 
        tol=0.00001, max_iter=5, early_stopping=False)
    clf_nn_D = MLPClassifier(hidden_layer_sizes=(50,50), verbose=False, 
        tol=0.00001, max_iter=4, early_stopping=False)
    # clf_rf = RandomForestClassifier(n_estimators=1000)

    clfs = [clf_nn_A for x in range(0, 28)]
    clfs += [clf_nn_B for x in range(0, 20)]
    clfs += [clf_nn_C for x in range(0, 6)]
    clfs += [clf_nn_D for x in range(0, 4)]
    save_name = Ensemble_NN_RF(clfs)

    clf_to_prediction(save_name, 2008)

    analyze_ensemble(save_name)
