#!/usr/bin/env python3

import numpy as np
import time
import csv

if __name__=='__main__':
    start = time.time()
    train_data = np.genfromtxt('../data/train_2008.csv', delimiter=',')
    end = time.time()

    time1 = end-start
    print(time1)

    start2 = time.time()
    with open('../data/train_2008.csv','r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]
    data_array = np.asarray(data)
    end2 = time.time()

    time2 = end2-start2
    print(time2)