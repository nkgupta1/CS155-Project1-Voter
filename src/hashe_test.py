#!/usr/bin/env python3

from common import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np

if __name__=='__main__':
    X, Y = train_2008_categorized()
    f = RandomForestClassifier(n_estimators=1000, n_jobs=2)
    f.fit(X, Y)

    X_test, id_s = test_2008_categorized()

    Y_predict = f.predict(X_test)

    res = format_results(id_s, Y_predict)
    with open('submissions/2008-rf.csv', 'w') as g:     
        g.write(res)
