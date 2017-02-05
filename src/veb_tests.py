import numpy as np
import time
from common import *
#import keras
#import sklearn

#print(train_data.shape, train_data.dtype)

x, y = import_train('../data/train_2008.csv')
x = import_test('../data/test_2008.csv')

# veb will start out with some svm, linear regression, clustering, and stuff


# final submission should have categorical labels 1, 2, 3, etc. as strings, not floats