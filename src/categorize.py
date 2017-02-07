#!/usr/bin/env python3

from common import *
from sklearn import svm
import pickle

def to_categorical(lst):
    """
    Maps lst into categories spread over different dimensions.
    Also returns a mapping array that can be used to replicate.
    """
    categories =  np.unique(lst, return_inverse=True)

    # create a map of the mappings used
    mapping = {}
    for i in range(len(categories[0])):
        mapping[categories[0][i]] = i

    # preallocate the array
    max_val = np.max(categories[1]) + 1
    categorized_values = np.zeros((max_val, len(lst)))

    # categorize
    for i in range(len(categories[1])):
        categorized_values[categories[1][i]][i] = 1

    return categorized_values, mapping

# data to be parsed
X_train_2008, Y_train_2008 = train_2008()
test_2008, ids_2008 = test_2008()
test_2012, ids_2012 = test_2012()

descriptions = []
count = 0

# VERY MAGIC NUMBERS
NUM_OLD_DIM = 381
NUM_NEW_DIM = 4056

# Parse the descriptions file.
with open('../label_descriptions.txt','r') as dest_f:
    label_name = True
    for line in dest_f:
        # skip over all the label names in the file
        if label_name:
            label_name = False
            continue

        # remove extra white space
        descriptions.append(line.strip())

        # set up for next iteration
        label_name = True
        count += 1

# Make sure file has expected number of dimensions
assert count == NUM_OLD_DIM

# make inner dimension of numpy array a single dimension of the data
# each row is a different dimension in the data
# each column is now one data point
X_train_2008 = X_train_2008.T

count = 0
# preallocate the array using a magic number to preserve memory
X_train_2008_fixed = np.empty( (NUM_NEW_DIM,len(X_train_2008[0])) )
# list of mappings so we can save them
mappings = []

# loop over all dimensions of the data
for i in range(len(X_train_2008)):
    # types of descriptions:
    # category, binary, binary/refused
    # integer
    label_type = descriptions[i]

    if label_type == 'delete':
        # want to delete the label
        pass
    elif label_type == 'integer':
        # normalize to [0,1]
        min_val = np.min(X_train_2008[i])
        max_val = np.max(X_train_2008[i])

        X_train_2008_fixed[count] = (X_train_2008[i] - min_val) / (max_val - min_val)
        # only added one new dimension to the new array
        count += 1
    else:
        # category, binary, binary/refused

        # returns the categorized and the mappings used
        categorized, mapping = to_categorical(X_train_2008[i])

        # save the mappings
        mappings.append(mapping)

        # put categorized data into preallocated numpy array
        for j in range(len(categorized)):
            X_train_2008_fixed[count] = categorized[j]
            count += 1

# make sure we used the expected number of dimensions in the numpy array
assert count == NUM_NEW_DIM

# save the mappings
pickle.dump(mappings, open('../data/category_mappings.pkl', 'wb'))