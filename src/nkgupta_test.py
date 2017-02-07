#!/usr/bin/env python3

from common import *
from sklearn import svm

def to_categorical(lst):
    categories =  np.unique(lst, return_inverse=True)
    max_val = np.max(categories) + 1
    to_ret = np.zeros((max_val, len(lst)))

    for i in range(len(categories)):
        to_ret[categories[i]][i] = 1

    return to_ret

X_train_2008, Y_train_2008 = train_2008()
test_2008, ids_2008 = test_2008()
test_2012, ids_2012 = test_2012()

descriptions = []
# VERY MAGIC NUMBER
NUM_DIM = 4056


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

count = 0
X_train_2008_fixed = np.empty( (NUM_DIM,len(X_train_2008[0])) )

for i in range(len(X_train_2008)):
    # print(i)
    # types of descriptions:
    # category, binary, binary/refused
    # integer
    label_type = descriptions[i]
    if label_type == 'delete':
        pass
    elif label_type == 'integer':
        # normalize to [0,1]
        min_val = np.min(X_train_2008[i])
        max_val = np.max(X_train_2008[i])

        X_train_2008_fixed[count] = (X_train_2008[i] - min_val) / (max_val - min_val)
        count += 1
    else:
        # category, binary, binary/refused
        # not finished, need to get around fixed size of numpy arrays
        # also implement to_categorical
        categorized = to_categorical(X_train_2008[i])
        for j in range(len(categorized)):
            X_train_2008_fixed[count] = categorized[j]
            count += 1

# print(X_train_2008_fixed.shape)
print(count)
assert count == NUM_DIM