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
        predictions = X
        for d in range(depth):
            predictions = self._run_layer(d, predictions)

        # Aggregated predictions
        final_predictions = mode(predictions)

        return final_predictions

    def retrain(self, X, Y, depth=5):
        """
        Replaces the current forest (if present) with a new one,
        trained to the specified depth and parameters.

        Arguments:
            X: input data
            Y: target predictions
            depth: number of layers trained 
        """

        # Check dimensions of input arrays
        assert X.shape[0] == Y.shape[0] # ???


        # Input valid. Go ahead and destroy any random forest already present
        # in the folder.
        for subdir, dirs, files in os.walk(self.folder):
            for file in files:
                if 'forest' in file:
                    os.remove(os.path.join(subdir, file))
                elif 'parameters' in file:
                    os.remove(os.path.join(subdir, file))

        # Translate window into a form more easily used by the program
        # =number of steps in each direction. Also translate 
        # intra_information, real_distances, and accessibility into 
        # numeric form for writing parameters file.
        window = int((window - 1) / 2)
        if intra_information == True:
            intra_information = 1
        else:
            intra_information = 0
        if real_distances == True:
            real_distances = 1
        else:
            real_distances = 0
        if accessibility == True:
            accessibility = 1
        else:
            accessibility = 0

        # Set parameters
        self.window = window
        self.intra_information = intra_information
        self.accessibility = accessibility
        self.real_distances = real_distances

        # Write to a parameters file
        f = open(os.path.join(self.folder, 'parameters.txt'), 'w')

        f.write('# Parameters file. Parameters set at forest creation, ' +
                'cannot be changed.\n\n')
        f.write('window : {0}\n'.format(window))
        f.write('intra_information : {0}\n'.format(intra_information))
        f.write('real_distances : {0}\n'.format(real_distances))
        f.write('accessibility : {0}\n'.format(accessibility))

        f.close()

        # Will need to modify contacts. Make a copy so that the original
        # ones aren't changed.
        c_list = deepcopy(contacts_list)

        # Sanitize data by aligning to pdb. Go ahead and find accessibility 
        # data, because that only needs to be found once. Same with real
        # distances.
        s_accessibility_list = []
        s_distances_list = []

        for i in range(len(pdb_list)):
            # Renumber all contacts to match pdb
            if c_list[i].get_pdb() != pdb_list[i]:
                c_list[i].renumber_to_pdb(seq_list[i], pdb_list[i])

            # Create a copy of the contact object, which will be used to store
            # actual distances.
            new_contact_obj = c_list[i].copy()

            # For each complex, go ahead and find accessibility. This only
            # needs to be found once, so it's better to find it now then
            # for every layer.
            if accessibility:
                s = c_list[i]._parse_structure(pdb_list[i])
                accessibility_list = self._check_accessibility(s)
            else:
                accessibility_list = []

            # Set distances
            if intra_information and real_distances:
                self._set_intra_distances(new_contact_obj, pdb_list[i])
            self._set_inter_distances(new_contact_obj, pdb_list[i])

            # Other data changes between layers, and will have to be found
            # at each depth. Go ahead and save this data to (very large!) 
            # lists.
            s_accessibility_list.append(accessibility_list)
            s_distances_list.append(new_contact_obj)

        # Train depth many layers
        for d in range(depth):
            X, Y, Z = [], [], []
            for i in range(len(pdb_list)):
                # Distances and accessibilities were already found earlier.
                # Now, get data, targets, and identifiers. Identifiers (Z)
                # allow us to modify data later on.
                X_raw, Y_raw, Z_raw = self._format(contact_distance, 
                                                   s_distances_list[i], 
                                                   c_list[i], 
                                                   s_accessibility_list[i])
                # Update actual lists now
                X.extend(X_raw)
                Y.extend(Y_raw)
    
                # Z is used for editing data afterwards. More helpful to append,
                # so that we can keep seperate lists for each pdb
                Z.append(Z_raw)

            # Now, we have valid data. Train and receive predictions
            predictions = self._train_layer(d, X, Y, n_estimators)

            # Modify data; specifically, change contact maps
            j = 0
            for k, contact_list in enumerate(Z):
                for contact in contact_list:
                    # Predictions are in the same order as contacts were entered.
                    # Since Z is in lists for each Contact, we can set all of 
                    # its values tp be the new predictions. These are then used
                    # in the next layer.
                    c_list[k].set((contact[0], contact[1], predictions[j]))
                    j += 1

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