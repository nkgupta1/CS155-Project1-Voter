#!/usr/bin/env python3

from common import *
from sklearn import svm
from sklearn.externals import joblib

def train_svm():
    """
    Train SVM on full data set.
    """
    X_train_2008, Y_train_2008 = train_2008()
    clf = svm.SVC()
    clf.fit(X_train_2008, Y_train_2008)
    joblib.dump(clf, 'saved_models/svc-default.jl')
    print(clf.score(X_train_2008, Y_train_2008))
    
def test_svm():
    print('hi')
    clf = joblib.load('saved_models/svc-default.jl')
    print('hi')
    X, _ = test_2008()
    print(X.shape)
    print(sum(clf.predict(X[4500:4800])))
    
test_svm()