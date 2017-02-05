#!/usr/bin/env python3

import numpy as np
import time
from common import *
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.externals import joblib

X_train, Y_train = train_2008()

#for arr in [X_train, Y_train]:
#	print arr.shape, arr.dtype

# normalize data
mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
std[std == 0] = 1  # in case values are constant for entire column
X_train = (X_train - mean) / std
#X_test = (X_test - mean) / std
print(Y_train.shape, Y_train.min(), Y_train.max())
#X_train, Y_train = X_train[:2000], Y_train[:2000]

in_sample_accuracy = lambda clf: np.mean(clf.predict(X_train) == Y_train)

def SVM(kernel='linear', cv=True, frac_train=False):
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
	clf = SVC(kernel=kernel, verbose=False)
	if cv:
		scores = cross_val_score(clf, X, Y, cv=5, 
			scoring='accuracy', verbose=True)
		print('cross-validation accuracy:', np.mean(scores))
	else:
		clf.fit(X, Y)
		finish = time.time()
		# print np.mean(clf.predict(X_train)), np.mean(Y_train)
		save_name = 'a-SVM_' + kernel + '-' + str(in_sample_accuracy(clf)) + '-'
		save_name += str(frac_train) + '-' + str(int(time.time() - start))
		joblib.dump(clf, 'saved_models/' + save_name)
	print('finished in', (time.time() - start), 'seconds')

	
def save_prediction(name, clf, year=2008):
	print('saving', name)
	if year == 2008:
		X_test, id_test = test_2008()
	elif year == 2012:
		X_test, id_test = test_2012()
	result = format_results(id_test, clf.predict(X_test))
	with open('submissions/' + str(year) + '-' + name + '.csv', 'w') as f:
		f.write(result)

def clf_to_prediction(year, name):
	save_prediction(name, joblib.load('saved_models/' + name), year)



SVM(kernel='linear', cv=False, frac_train=0.2)

#clf_to_prediction(2008, 'a-SVM_linear-0.762351740455-0.1-124')


























