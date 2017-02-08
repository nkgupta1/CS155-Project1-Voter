#!/usr/bin/env python3

from common import *
import numpy as np
import time
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def Ensemble_NN_RF(clfs):
    X_train, Y_train = train_2008_categorized()
    start = time.time()
    clf = VotingClassifier([(str(i), clfs[i]) for i in range(len(clfs))], 
            voting='soft', n_jobs=2)
    clf.fit(X_train, Y_train)
    score = clf.score(X_train, Y_train)
    print('score:', score)
    save_name = 'e-Ensemble-' + str(score) + '-' + str(int(time.time() - start))
    joblib.dump(clf, 'saved_models/' + save_name)
    print('finished in', (time.time() - start), 'seconds')

    return save_name



if __name__=='__main__':
    clf_nn = MLPClassifier(hidden_layer_sizes=(100,50), verbose=True, 
        tol=0.00001, max_iter=5, early_stopping=True)
    clf_rf = RandomForestClassifier(n_estimators=100)

    clfs = [clf_nn, clf_nn, clf_nn, clf_nn, clf_nn, clf_nn,
            clf_rf, clf_rf, clf_rf, clf_rf, clf_rf]
    save_name = Ensemble_NN_RF(clfs)

    clf_to_prediction(save_name, 2008)

    # analyze_ensemble('d-MLP-(50, 50)-4-0.888737686919-384', prnt=True)