#!/usr/bin/env python3

from common import import_train

if __name__=='__main__':
    X, Y = import_train("../data/train_2008.csv")
    print(X.shape)
    print(Y.shape)