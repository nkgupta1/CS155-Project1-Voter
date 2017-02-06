#!/usr/bin/env python3

from common import *
from sklearn import svm


X_train_2008, Y_train_2008 = train_2008()
test_2008, ids_2008 = test_2008()
test_2012, ids_2012 = test_2012()

print (X_train_2008[0])