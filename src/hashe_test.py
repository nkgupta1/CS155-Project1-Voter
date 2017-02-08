#!/usr/bin/env python3

from common import *
import numpy as np

if __name__=='__main__':
    # Caveat; need classes to be zero-indexed
    f = forest('saved_models/test_forest')
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    Y = np.array([0,1,1,0])
    f.fit(X,Y,depth=3,n_estimators=30)
    fp = f.predict(X)[1].astype(int)
    print(fp)
    # X, Y = train_2008_categorized()
    # print('Length Y = {0}'.format(len(Y)))
    # print('Num 1 = {0}'.format(sum(Y)))
    # print('Num 0 = {0}'.format(len(Y)-sum(Y)))
    # print('Frac 1 = {0}'.format(sum(Y)/len(Y)))