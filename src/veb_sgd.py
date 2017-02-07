#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from veb_common import *
from sklearn.model_selection import KFold

threads = 2

X_train, Y_train = normalized_data()

def SGD(a=1., n_iter=5, cv=True):
    '''
    param: cv: if cv is false, does not cross validate, trains on full data, 
    and pickles the result.
    param: frac_train: fraction of training to use for training (if False, 
    use all of the training data)
    '''
    print('learning', n_iter, a, cv)
    start = time.time()
    X, Y = X_train, Y_train
    clf = SGDClassifier(verbose=False, n_iter=n_iter, n_jobs=-1, alpha=a)
    if cv:
        scores = cross_val_score(clf, X, Y, cv=5, 
            scoring='accuracy', verbose=True)
        print('cross-validation accuracy:', np.mean(scores))
    else:
        clf.fit(X, Y)
        finish = time.time()
        score = clf.score(X_train, Y_train)
        # print np.mean(clf.predict(X_train)), np.mean(Y_train)
        save_name = 'b-SGD_hinge-' + str(n_iter) + '-' + str(a) + '-'
        save_name += str(score) + '-' + str(int(time.time() - start))
        joblib.dump(clf, 'saved_models/' + save_name)
    print('finished in', (time.time() - start), 'seconds')


#SGD(n_iter=3000, a=0.0005, cv=False)



def plot_epoch_dependence():
    from matplotlib import pyplot as plt
    in_errs, out_errs = [], []
    kf = KFold(n_splits=5, shuffle=True)
    epochs = np.arange(1000, 2100, 500)
    print('epochs:', epochs)
    for e in epochs:
        clf = SGDClassifier(alpha=0.0005, n_iter=e)
        validation_error = []
        for train_index, test_index in kf.split(X_train):
            clf.fit(X_train[train_index], Y_train[train_index])
            validation_error.append(1.0 - clf.score(X_train[test_index], Y_train[test_index]))
        print e, np.mean(validation_error)
        out_errs.append(np.mean(validation_error))
        clf.fit(X_train, Y_train)
        score = 1.0 - clf.score(X_train, Y_train)
        print e, score
        in_errs.append(score)
    print('errors:', in_errs, out_errs)
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.title('SGD: Error vs. Epoch')
    plt.plot(epochs, np.array(in_errs), label='In-Sample Error')
    plt.plot(epochs, np.array(out_errs), label='5-Fold CV Error')
    plt.legend()#loc='center right')
    plt.savefig('saved_models/sgd-vs-epoch', bbox_inches='tight')


plot_epoch_dependence()


def plot_alpha_dependence():
    from matplotlib import pyplot as plt
    in_errs, out_errs = [], []
    kf = KFold(n_splits=5, shuffle=True)
    alphas = np.arange(0.0001, 0.0007, 0.00004)
    print('alphas', alphas)
    for alpha in alphas:
        clf = SGDClassifier(alpha=alpha, n_iter=2500)
        validation_error = []
        for train_index, test_index in kf.split(X_train):
            clf.fit(X_train[train_index], Y_train[train_index])
            validation_error.append(1.0 - clf.score(X_train[test_index], Y_train[test_index]))
        print alpha, np.mean(validation_error)
        out_errs.append(np.mean(validation_error))
        clf.fit(X_train, Y_train)
        score = 1.0 - clf.score(X_train, Y_train)
        print alpha, score
        in_errs.append(score)
    print in_errs, out_errs
    print('errors:', in_errs, out_errs)
    plt.xlabel('Alpha')
    plt.ylabel('Classification Error')
    plt.title('SGD: Error vs. Alpha')
    print alphas.shape, np.array(in_errs).shape
    plt.plot(alphas, np.array(in_errs), label='In-Sample Error')
    plt.plot(alphas, np.array(out_errs), label='5-Fold CV Error')
    plt.legend()#loc='center right')
    plt.savefig('saved_models/sgd-vs-alpha', bbox_inches='tight')

#plot_alpha_dependence()