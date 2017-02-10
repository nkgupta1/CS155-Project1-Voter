#!/usr/bin/env python3

from common import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score

if __name__=='__main__':
    X_train, Y_train = train_2008_categorized()
    f = RandomForestClassifier(n_estimators=1000, n_jobs=2)
    # f.fit(X_train, Y_train)
    scores = cross_val_score(f, X_train, Y_train, cv=5, scoring='accuracy', verbose=True)
    score = np.mean(scores)
    print('cross-validation accuracy:', score)

    # X_test, id_s = test_2008_categorized()

    # Y_predict = f.predict(X_test)

    # res = format_results(id_s, Y_predict)
    # with open('submissions/2008-rf.csv', 'w') as g:     
    #     g.write(res)
