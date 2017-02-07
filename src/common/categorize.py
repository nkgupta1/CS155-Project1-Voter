#!/usr/bin/env python3

import common
from sklearn import svm
import numpy as np
import pickle
from sklearn.externals import joblib

def to_categorical(lst, mapping=None, test=False):
    """
    Maps lst into categories spread over different dimensions.
    Also returns a mapping array that can be used to replicate.
    """
    if not test:

        categories = np.unique(lst)

        # create a map of the mappings used
        mapping = {}
        for i in range(len(categories)):
            mapping[categories[i]] = i

    # preallocate the array
    max_val = len(mapping)
    categorized_values = np.zeros((max_val, len(lst)))

    # categorize
    for i in range(len(lst)):
        a = lst[i]
        try:
            # catches the case where a data value is seen in the test set that
            # isn't in the training set. happens about 90000 times between the
            # 2 data sets out of a total of (16000+82000)*4000=392000000 points
            # or 0.02% but most of these cases are pretty trivial
            b = mapping[a]
            categorized_values[b][i] = 1
        except Exception as e:
            # ignore the cases with unrecognized values
            continue

    return categorized_values, mapping

def categorize_data(data, descriptions, mappings=None, test=False):
    """
    Categorizes the input data according to descriptions
    Mappings represents the mappings we are to use, such as in the case of test 
    data
    """

    # VERY MAGIC NUMBER!!!
    NUM_NEW_DIM = 4054

    # Make sure file has expected number of dimensions
    assert len(descriptions) == len(data[0])

    # make each row is a different dimension in the data
    # make each column is now one data point
    data = data.T

    count = 0
    # preallocate the array using a magic number to preserve memory
    data_fixed = np.empty( (NUM_NEW_DIM, len(data[0])) )

    if test:
        assert len(descriptions) == len(mappings)
    else:
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
            if not test:
                mappings.append(None)
        elif label_type == 'integer':
            # normalize to [0,1]
            
            min_val = max_val = 0;
            
            if test:
                min_val = mappings[i][0]
                max_val = mappings[i][1]
            else:
                min_val = np.min(data[i])
                max_val = np.max(data[i])
                # save the mappings
                mappings.append((min_val, max_val))

            data_fixed[count] = (data[i] - min_val) / (max_val - min_val)
            # only added one new dimension to the new array
            count += 1

        else:
            # category, binary, binary/refused
            categorized = None

            if test:
                categorized, _ = to_categorical(data[i], mappings[i], test=True)
            else:
                # returns the categorized and the mappings used
                categorized, mapping = to_categorical(data[i])

                # save the mappings
                mappings.append(mapping)
                


            # put categorized data into preallocated numpy array
            for j in range(len(categorized)):
                data_fixed[count] = categorized[j]
                count += 1


    # make sure we used the expected number of dimensions in the numpy array
    assert count == NUM_NEW_DIM

    # make each row one data point
    # each column one data dimension
    return data_fixed.T, mappings

def parse_descriptions():
    """
    Parse the descriptions.
    A dimension contains the dimension label and then the type of dimension on 
    the next line
    """
    descriptions = []

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
    return descriptions
    

def save_categorized_data():
    """
    needs to be run from src.
    """
    # data to be parsed
    X_train_2008, Y_train_2008 = common.train_2008()
    test_2008, ids_2008 = common.test_2008()
    test_2012, ids_2012 = common.test_2012()

    descriptions = parse_descriptions()

    X_train_2008_fixed, mappings = categorize_data(X_train_2008, descriptions)
    X_test_2008_fixed, _ = categorize_data(test_2008, descriptions, mappings, test=True)
    X_test_2012_fixed, _ = categorize_data(test_2012, descriptions, mappings, test=True)

    joblib.dump(X_train_2008_fixed, '../data/X_train_2008_categorized.jbl')
    joblib.dump(X_test_2008_fixed, '../data/X_test_2008_categorized.jbl')
    joblib.dump(X_test_2012_fixed, '../data/X_test_2012_categorized.jbl')

def train_2008_categorized():
    """
    Gets data and labels from the 2008 training set categorized
    Must be run from src
    """
    return joblib.load('../data/X_train_2008_categorized.jbl'), joblib.load('../data/Y_train_2008.jbl')

def test_2008_categorized():
    """
    Gets data from the 2008 test set categorized
    Must be run from src
    """
    return joblib.load('../data/X_test_2008_categorized.jbl'), joblib.load('../data/id_test_2008.jbl')

def test_2012_categorized():
    """
    Gets data from the 2012 test set categorized
    Must be run from src
    """
    return joblib.load('../data/X_test_2012_categorized.jbl'), joblib.load('../data/id_test_2012.jbl')