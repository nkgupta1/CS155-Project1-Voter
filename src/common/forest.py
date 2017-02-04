#!/usr/bin/env python3

from copy import deepcopy
from math import sqrt
from common import mode

import os
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib
except ImportError:
    raise ImportError('[!] Forest requires Scikit-Learn.')

class ForestError():
    """
    Exceptions raised from errors involving random forests.
    """
    
    pass

class forest():
    """
    Random forests used to analyze voting patterns.
    """

    def __init__(self, folder):
        """
        Creates a forest object that stores/reads a forest present 
        in the given folder. A forest must be present before other
        methods may be used.

        Arguments:
            folder: The location of a forest, or a folder in which
                to place a trained forest.
        """

        # Load in folder
        self.load(folder)

    def load(self, folder):
        """
        Checks the forest. Can also be used to load in a new forest.

        Arguments:
            folder: The location of a forest, or a folder in which
                to place a trained forest.
        """

        # Check that the folder exists or can be made
        if not os.path.isdir(folder):
            try:
                os.makedirs(folder)
            except OSError:
                raise ForestError('Could not create random forest folder: {0}'
                                  .format(folder))

        # Find the maximum depth within the folder. If no file, depth is 0.
        max_depth = 0
        while(os.path.isfile(os.path.join(folder, 'forest_{0}.dat'
                                          .format(max_depth)))):
            max_depth += 1

        # Reset the depth and folder
        self.depth = max_depth
        self.folder = folder

        return True

    def predict(self, X, depth=-1):
        """
        Predicts voting turnout from previously trained forest.

        Arguments:
            X: input data vector
            depth: optional flag for training only on the first d layers
        """

        # A forest must exist before this method can be called
        self.load(self.folder)
        if self.depth == 0:
            raise ForestError('[!] No forest was found at {0}'
                              .format(self.folder) +
                              '. Please train or load a forest.')

        # If depth wasn't set, use max depth
        if depth == -1:
            depth = self.depth

        # If depth is larger than possible raise error
        if depth > self.depth:
            raise ForestError('[!] Forest of depth {0} cannot predict '
                              .format(self.depth) +
                              'to depth {0}'.format(depth))

        # Depth must be positive...
        if depth <= 0:
            raise ForestError('[!] Depth must be positive integer. Depth ' +
                              '{0} is not positive.'.format(depth))

        # ...and also an integer
        try:
            depth = int(depth)
        except:
            raise ForestError('[!] Depth must be positive integer. Depth ' +
                              '{0} is not an integer.'.format(depth))

        # Run on each layer and update results.
        predictions = X # X isn't predictions, but is overwritten by predictions
        for d in range(depth):
            predictions = self._run_layer(d, predictions)

        # Aggregated predictions
        final_predictions = mode(predictions)

        return final_predictions

    def fit(self, X, Y, depth=5, n_estimators=100):
        """
        Replaces the current forest (if present) with a new one,
        trained to the specified depth and parameters.

        Arguments:
            X: input data
            Y: target predictions
            depth: number of layers trained 
        """

        # Check dimensions of input arrays
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape == 0)


        # Input valid. Go ahead and destroy any random forest already present
        # in the folder.
        for subdir, dirs, files in os.walk(self.folder):
            for file in files:
                if 'forest' in file:
                    os.remove(os.path.join(subdir, file))
                elif 'parameters' in file:
                    os.remove(os.path.join(subdir, file))

        # Train depth many layers
        predictions = X # X not predictions, overwritten by predictions
        for d in range(depth):
            predictions = self._train_layer(d, predictions, Y, n_estimators)

        return True

    def _run_layer(self, depth, X):
        """
        Runs a single layer of a random forest.

        Arguments:
            depth: The current depth of the random forest.
            X: The input parameters.
        """

        # Load forest from compressed files
        model = joblib.load(os.path.join(self.folder, 'forest_{0}.dat'.format(depth)))
        
        # Find predictions for each data point
        new_predicted = []
        for x in X:
            predicted = [tree.predict(X) for tree in model.estimators_]
            # CHECK IF INTS, BOOLS, WHATEVER
            new_predicted.append(predicted)
        new_predicted = np.asarray(new_predicted)
    
        # print('[+] Layer {0} of random forest has predicted predictions.'.format(depth + 1))
    
        return new_predicted

    def _train_layer(self, depth, X, Y, n_estimators):
        """
        Trains a single layer of a random forest.

        Arguments:
            depth: The current depth of the random forest.
            X: The input parameters.
            Y: The targets.
            n_estimators: The number of trees in the forest.
        """

        # START HERE

        # Fit the first layer
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X, Y)
    
        # Since we want a binary classification anyway, we can just take these probabilities across
        # layers. Then, we feed that as input to the next layer
        predicted = model.predict_proba(X)
    
        # First model trained. Save in compressed format (joblib preferable to pickle)
        joblib.dump(model, os.path.join(self.folder, 'forest_{0}.dat'.format(depth)))

        print('[+] Layer {0} of random forest has been trained.'.format(depth + 1))
    
        return [i[1] for i in predicted]