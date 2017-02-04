#!/usr/bin/env python3

from common import *
from sklearn.externals import joblib

# save_parsed_data()

X, Y = test_2012()

print(X.shape)
print(Y.shape)

# joblib.dump(X, '../data/X_test_2008.pkl')

# X_train_2008 = joblib.load('saved_models/X_train_2008.pkl')
# Y_train_2008 = joblib.load('saved_models/Y_train_2008.pkl')
