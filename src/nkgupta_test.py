#!/usr/bin/env python3

from common import *
from sklearn import svm
from keras.utils.np_utils import to_categorical


X_train_2008, Y_train_2008 = train_2008()
test_2008, ids_2008 = test_2008()
test_2012, ids_2012 = test_2012()

descriptions = []


with open('../label_descriptions.txt','r') as dest_f:
    count = 0
    label_name = True
    for line in dest_f:
        if label_name:
            label_name = False
            continue
        descriptions.append(line.strip())

        label_name = True
        count += 1


X_train_2008 = X_train_2008.T

assert len(X_train_2008) == len(descriptions)

for i in range(len(X_train_2008)):
    # types of descriptions:
    # category, binary, binary/refused
    # integer
    label_type = descriptions[i]
    if label_type == 'integer':
        # normalize to [0,1]
        min_val = np.min(X_train_2008[i])
        max_val = np.max(X_train_2008[i])
        X_train_2008[i] = (X_train_2008[i] - min_val) / (max_val - min_val)
    else:
        # category, binary, binary/refused
        # not finished, need to get around fixed size of numpy arrays
        # also implement to_categorical
        print(X_train_2008[i])
        print(to_categorical(X_train_2008[i]))
        print(X_train_2008[i])
        quit()

X_train_2008[0] = 1
print(len(X_train_2008[0]))