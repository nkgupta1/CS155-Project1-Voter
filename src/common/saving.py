import numpy as np
from sklearn.externals import joblib
from categorize import *
from common import *

def save_prediction(name, clf, year, categorized):
    print('saving', name)
    id_test, X_test = categorized_test_data(year, categorized)
    result = format_results(id_test, clf.predict(X_test).astype(np.uint8))
    with open('submissions/' + str(year) + '-' + name + '.csv', 'w') as f:
        f.write(result)

def clf_to_prediction(name, year=2008, categorized=True):
    save_prediction(name, joblib.load('saved_models/' + name), year, categorized)

def categorized_test_data(year, categorized, normed=False):
    if categorized:
        X_train, Y_train = train_2008_categorized()
    else:
        X_train, Y_train = train_2008()
    if year == 2008:
        if categorized:
            X_test, id_test = test_2008_categorized()
        else:
            X_test, id_test = test_2008()
    elif year == 2012:
        if categorized:
            X_test, id_test = test_2012_categorized()
        else:
            X_test, id_test = test_2012()
    if normed:
        mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
        std[std == 0] = 1
        X_test = (X_test - mean) / std
    return id_test, X_test

def analyze_ensemble(clf_name, year=2008, categorized=True, prnt=True):
    id_test, X_test = categorized_test_data(year, categorized)
    eclf = joblib.load('saved_models/' + clf_name)
    predictions = []
    for clf in eclf.estimators_:
        predictions.append(clf.predict(X_test))
    predictions = np.stack(predictions, axis=1)
    if prnt:
        for row in predictions:
            print row
    return predictions