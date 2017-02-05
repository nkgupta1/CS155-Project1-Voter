#!/usr/bin/env python3

from common import *
from sklearn import svm
from sklearn.externals import joblib
import numpy as np

def train_svm():
    """
    Train SVM on full data set.
    """
    X_train_2008, Y_train_2008 = train_2008()

    mapping = {0:-1, 1:1}
    Y_train_2008 = np.vectorize(mapping.get)(Y_train_2008)

    clf = svm.SVC(cache_size=8000)
    clf.fit(X_train_2008, Y_train_2008)
    joblib.dump(clf, 'saved_models/svc-default-vect.jl')
    print(clf.score(X_train_2008, Y_train_2008))
    
def test_svm():
    clf = joblib.load('saved_models/svc-default.jl')
    X, _ = test_2008()
    print(X.shape)
    Y = clf.predict(X)
    print(sum(Y))
    np.savetxt('saved_models/svm-default-vect-2008-predict.txt', Y)
    
    
train_svm()
test_svm()
