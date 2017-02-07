#!/usr/bin/env python3

import common
from sklearn import svm
import numpy as np
import pickle
from sklearn.externals import joblib

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

def categorize_data():
    """
    Categorizes the input data according to label_descriptions.txt
    """

    # data to be parsed
    X_train_2008, Y_train_2008 = common.train_2008()
    test_2008, ids_2008 = common.test_2008()
    test_2012, ids_2012 = common.test_2012()

    descriptions = []

    # VERY MAGIC NUMBER!!!
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

    # Make sure file has expected number of dimensions
    assert len(descriptions) == len(X_train_2008[0])

    # make each row is a different dimension in the data
    # make each column is now one data point
    X_train_2008 = X_train_2008.T

    count = 0
    # preallocate the array using a magic number to preserve memory
    X_train_2008_fixed = np.empty( (NUM_NEW_DIM, len(X_train_2008[0])) )

    # list of mappings so we can save them
    mappings = []

    # loop over all dimensions of the data
    for i in range(len(descriptions)):
        # types of descriptions:
        # category, binary, binary/refused
        # integer
        label_type = descriptions[i]

        if label_type == 'delete':
            # want to ignore this label
            pass
        elif label_type == 'integer':
            # normalize to [0,1]
            min_val = np.min(X_train_2008[i])
            max_val = np.max(X_train_2008[i])

            # save the mappings
            mappings.append((min_val, max_val))

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

    # make each row one data point
    # each column one data dimension
    categorized = categorized.T

    # make sure we used the expected number of dimensions in the numpy array
    assert count == NUM_NEW_DIM

    print (mappings)

    # map the test data sets now
    categorize_test(test_2008, descriptions, mappings, NUM_NEW_DIM)

    # save the results
    pickle.dump(mappings, open('../data/category_mappings.pkl', 'wb'))
    joblib.dump(categorized, '../data/X_train_2008_categorized.jbl')

def categorize_test(test, descriptions, mappings, NUM_NEW_DIM):


    # make each row is a different dimension in the data
    # make each column is now one data point
    test = test.T

    count = 0
    # preallocate the array using a magic number to preserve memory
    test_fixed = np.empty( (NUM_NEW_DIM, len(test[0])) )

    # loop over all dimensions of the data
    for i in range(len(descriptions)):
        # types of descriptions:
        # category, binary, binary/refused
        # integer
        label_type = descriptions[i]

        if label_type == 'delete':
            # want to ignore this label
            pass
        elif label_type == 'integer':
            # normalize to [0,1]
            min_val = mappings[count][0]
            max_val = mappings[count][1]

            test_fixed[count] = (test[i] - min_val) / (max_val - min_val)
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


categorize_data()