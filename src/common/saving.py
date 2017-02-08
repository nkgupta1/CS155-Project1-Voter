import numpy as np
from sklearn.externals import joblib
from categorize import *

def save_prediction(name, clf, year, categorized):
    if categorized:
        X_train, Y_train = train_2008_categorized()
    else:
        X_train, Y_train = train_2008()
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    std[std == 0] = 1
    print('saving', name)
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
    X_test = (X_test - mean) / std
    result = format_results(id_test, clf.predict(X_test).astype(np.uint8))
    with open('submissions/' + str(year) + '-' + name + '.csv', 'w') as f:
        f.write(result)

def clf_to_prediction(year, name, categorized=True):
    save_prediction(name, joblib.load('saved_models/' + name), year, categorized)
