#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from veb_common import normalized_data, save_prediction, clf_to_prediction 


threads = 1

X_train, Y_train = normalized_data()


def linear_regression(frac_train=False):
	'''
	param: cv: if cv is false, does not cross validate, trains on full data, 
	and pickles the result.
	param: frac_train: fraction of training to use for training (if False, 
	use all of the training data)
	'''
	print('starting')
	start = time.time()
	if frac_train:
		X = X_train[:int(frac_train * X_train.shape[0])]
		Y = Y_train[:int(frac_train * Y_train.shape[0])]
	else:  X, Y = X_train, Y_train
	clf = LinearRegression(n_jobs=threads)
	clf.fit(X, Y)
	finish = time.time()
	# print np.mean(clf.predict(X_train)), np.mean(Y_train)
	y_predict = clf.predict(X_train)
	y_predict[y_predict >= 0.5] = 1
	y_predict[y_predict != 1] = 0
	accuracy = np.mean(y_predict == Y_train)
	print('accuracy:', accuracy)
	save_name = 'a-LinearRegression' + '-' + str(accuracy) + '-'
	save_name += str(frac_train) + '-' + str(int(time.time() - start))
	joblib.dump(clf, 'saved_models/' + save_name)
	print('finished in', (time.time() - start), 'seconds')


linear_regression()

#clf_to_prediction(2008, 'a-SVM_linear-0.762351740455-0.1-124')


