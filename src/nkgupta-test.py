#!/usr/bin/env python3

from common import *
from sklearn import svm

X_train_2008, Y_train_2008 = train_2008()
clf = svm.SVC()
clf.fit(X_train_2008, Y_train_2008)
print(clf.score(X_train_2008, Y_train_2008))
