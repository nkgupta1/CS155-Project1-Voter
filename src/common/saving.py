import numpy as np
from sklearn.externals import joblib
from categorize import *

def save_prediction(name, clf, year, categorized):
    print('saving', name)
    id_test, X_test = normed_cat_test_data(year, categorized)
    result = format_results(id_test, clf.predict(X_test).astype(np.uint8))
    with open('submissions/' + str(year) + '-' + name + '.csv', 'w') as f:
        f.write(result)

def clf_to_prediction(name, year=2008, categorized=True):
    save_prediction(name, joblib.load('saved_models/' + name), year, categorized)
