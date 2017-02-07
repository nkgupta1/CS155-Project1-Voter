#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from veb_common import normalized_data, save_prediction, clf_to_prediction 


threads = 1

X_train, Y_train = normalized_data()

print 'hello world'

