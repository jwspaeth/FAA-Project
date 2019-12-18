
import os
import copy
import pickle
import math

import numpy as np
import pandas as pd


class FaaDataset():
    """
    Represents the faa data.
    """

    dataset_path = "data/faa-data/"
    test_file = "ou_test_data_8875846_35100.pickle"
    feature_length = 3
    shuffle_seed = 6 # This value needs to remain the same from run to run.

    def __init__(self, master_config = None):

        self.master_config = master_config

    def get_full_generator(self, batch_size=1):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        yield None

    def get_training_generator(self, batch_size=1, sample_weight_mode=None):
        training_inds, val_inds, feature_array, label_array = self._load_data()

        yield feature_array[training_inds], label_array[training_inds], None

    def get_validation_generator(self):
        training_inds, val_inds, feature_array, label_array = self._load_data()

        yield feature_array[val_inds], label_array[val_inds], None

    #Last three functions returns full dataset
    def get_training_features_and_labels(self):
        training_inds, val_inds, feature_array, label_array = self._load_data()

        return feature_array[training_inds], label_array[training_inds], None

    def get_validation_features_and_labels(self):
        training_inds, val_inds, feature_array, label_array = self._load_data()

        return feature_array[val_inds], label_array[val_inds], None

    def get_full_features_and_labels(self):
        training_inds, val_inds, feature_array, label_array = self._load_data()

        return feature_array, label_array

    def _get_sample_weights(self, model_config):
        return None

    def _load_data(self, val_size=.2):
        '''
        Performs both feature/label and training/validation splits
        One function because individual functions aren't generally used
        '''

        shuffled_indices, dataset = self._load_raw_data()

        training_inds, val_inds = self._train_split(indices=shuffled_indices, val_size=val_size)
        feature_segment_array, label_segment_array = self._feature_label_split(dataset)

        return training_inds, val_inds, feature_segment_array, label_segment_array

    def _load_raw_data(self):

        # Unpickle data file. All relevant files are currently .pickle files.
        unpickled = pickle.load(
            open(self.dataset_path + self.test_file, 'rb'))

        # Each flight is a different length of time, so it can't be stacked into a matrix.
        # This function unpacks the unpickled object into a list of numpy arrays, where each row is a numpy array,
        #   and each numpy array has the shape (data_length, 3). Each of these rows represents a different flight track
        #   (different samples)
        row_list = self._create_row_list(unpickled)

        # Split rows into segments.
        # Returns (features, labels)
        segment_stack = self._create_segments(row_list)

        # Create indices and shuffle for segments
        indices = list(range(0, segment_stack.shape[0]))
        np.random.seed(self.shuffle_seed)
        shuffled_indices = np.random.permutation(indices)

        return shuffled_indices, segment_stack

    def _create_row_list(self, unpickled_list):
        '''
        Each flight is a different length of time, so it can't be stacked into a matrix.
        This function unpacks the unpickled object into a list of numpy arrays, where each row is a numpy array,
            and each numpy array has the shape (data_length, 3). Each of these rows represents a different flight track
            (different samples)
        '''
        row_list = []
        for row in unpickled_list['data']:
            xyz_list = [
                row[11],
                row[12],
                row[13]
                ]
            data_length_stack = np.column_stack(xyz_list)
            row_list.append(data_length_stack)

        return row_list

    def _create_segments(self, row_list):
        '''
        Splits each row into segments of the desired length.
        Returns segment stack of all segments from all rows
        These array shapes are approximate, because if the data_length doesn't evenly divide by feature_length, the truncated
            values will be thrown out.
        '''

        # Split each row into segments
        segment_stack = []
        for row in row_list:
            row_segments = self._split_row(row)
            if row_segments is not None:
                segment_stack.append(row_segments)

        # Stack all segments together
        segment_stack = np.concatenate(segment_stack, axis=0)

        return segment_stack

    def _split_row(self, row):
        '''
        Splits an individual row into segments. For more details see _create_segments function.
        Returns array representing split segments, with shape (n_segments, feature_length+2, channels)
        If the input row isn't large enough to create any segments, return None.
        '''
        segment_list = []
        for i in range(len(row)//(self.feature_length+2)):
            begin = i * (self.feature_length+2)
            end = (i + 1) * (self.feature_length+2)

            segment_list.append(row[begin:end])

        if segment_list:
            segment_array = np.stack(segment_list, axis=0)
            return segment_array
        else:
            return None

    def _train_split(self, indices, val_size=.2):
        '''
        Splits dataset indices into training and validation splits
        '''

        # Create training split
        train_beg = 0
        train_end = math.floor(indices.shape[0]*(1-val_size))
        training_inds = indices[train_beg:train_end]

        # Create validation split
        val_beg = math.floor(indices.shape[0]*(1-val_size))
        val_end = indices.shape[0]
        val_inds = indices[val_beg:val_end]

        return training_inds, val_inds

    def _feature_label_split(self, dataset):
        '''
        Splits dataset into features and labels
        '''

        # Create feature array
        feature_segment_array = dataset[:, 1:self.feature_length+1, :]

        # Create label array
        past_point = dataset[:, 0, :]
        future_point = dataset[:, self.feature_length+1, :]
        label_segment_array = np.stack([past_point, future_point], axis=1)

        return feature_segment_array, label_segment_array









        