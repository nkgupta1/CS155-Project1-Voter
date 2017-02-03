#!/usr/bin/env python3

"""Commonly used functions, just so that we don't have to rewrite them
in every file.
"""

import numpy as np
import csv

def import_data(filename):
    """
    Imports data from provided csv file, returns as
    numpy array.
    """

    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]
    data = np.asarray(data)
    data = data.astype(np.float32)