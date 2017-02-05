#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.linear_model import Lasso, Ridge #, RidgeClassifierCV
from sklearn.externals import joblib
from veb_common import *
from sklearn.model_selection import KFold

X_train, Y_train = normalized_data()

in_sample_accuracy = lambda clf: np.mean(clf.predict(X_train) == Y_train)


def ridge_lasso_regression(m_type, tol=.0001, a=1., cv=True):
    '''
    param: m_type: 'ridge' or 'lasso' regression
    param: tol: tolerance value. see sklearn
    param: cv: if cv is false, does not cross validate, trains on full data, 
    and pickles the result.
    '''
    print('learning', m_type, tol, a, cv)
    start = time.time()
    X, Y = X_train, Y_train
    if m_type == 'ridge':
        clf = Ridge(solver='auto', tol=tol, alpha=a)
    elif m_type == 'lasso':
        clf = Lasso(tol=tol, alpha=a)
    if cv:
        scores = cross_val_score(clf, X, Y, cv=5, 
            scoring='accuracy', verbose=False)
        accuracy = np.mean(scores)
        print('cross-validation accuracy:', accuracy)
    else:
        clf.fit(X, Y)
        y_predict = clf.predict(X_train)
        y_predict[y_predict >= 0.5] = 1
        y_predict[y_predict != 1] = 0
        accuracy = np.mean(y_predict == Y_train)
        finish = time.time()
        # print np.mean(clf.predict(X_train)), np.mean(Y_train)
        save_name = 'c-' + m_type.upper() + '-' + str(tol) + '-' + str(a)
        save_name += '-' + str(accuracy) + '-' + str(int(time.time() - start))
        joblib.dump(clf, 'saved_models/' + save_name)
    print('finished in', (time.time() - start), 'seconds')
    return accuracy

# ridge_lasso_regression('ridge', cv=False)

def plot_alpha_dependence():
    from matplotlib import pyplot as plt

    in_errs, out_errs = [], []
    # Lasso: np.arange(0.001, .01, .001) # Ridge: np.arange(1, 10000, 1000)
    alphas = np.arange(0.001, .01, .001)
    print('alphas:', alphas)

    kf = KFold(n_splits=5, shuffle=True)
    for alpha in alphas:
        validation_error = []
        clf = Lasso(tol=0.00001, alpha=alpha)
        for train_index, test_index in kf.split(X_train):
            clf.fit(X_train[train_index], Y_train[train_index])
            y_predict = clf.predict(X_train[test_index])
            y_predict[y_predict >= 0.5] = 1
            y_predict[y_predict != 1] = 0
            validation_error.append(1.0 - np.mean(y_predict == Y_train[test_index]))
        out_errs.append(np.mean(validation_error))
        clf.fit(X_train, Y_train)
        y_predict = clf.predict(X_train)
        y_predict[y_predict >= 0.5] = 1
        y_predict[y_predict != 1] = 0
        in_errs.append(1.0 - np.mean(y_predict == Y_train))
    print('errors:', in_errs, out_errs)
    plt.xlabel('Alpha')
    plt.ylabel('Classification Error')
    plt.title('Lasso: Error vs. Alpha')
    plt.plot(alphas, np.array(in_errs), label='In-Sample Error')
    plt.plot(alphas, np.array(out_errs), label='5-Fold CV Error')
    plt.legend(loc='center right')
    plt.savefig('saved_models/lasso-vs-alpha', bbox_inches='tight')

plot_alpha_dependence()

