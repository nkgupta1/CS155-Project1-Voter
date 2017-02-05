#!/usr/bin/env python3

from common import *
from sklearn import svm


data_array = np.loadtxt('saved_models/svm-default-vect-2008-predict.txt', dtype=np.int32)
_, ids = test_2008()

with open('submissions/1.csv','w') as dest_f:
    dest_f.write(format_results(ids, data_array))