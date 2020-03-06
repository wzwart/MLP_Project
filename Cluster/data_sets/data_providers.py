    # -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""
from __future__ import print_function

DEFAULT_SEED = 20112018
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    pass
else:
    pass

    # from  torch import  FloatTensor
    from torchvision import transforms

    from data_sets.data_set_youtube import DatasetYoutube
from data_sets.data_set_youtube import Rescale, RandomCrop, Normalize,ToTensor


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, points, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        self.points = points
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.points = self.points[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]
        self.points = self.points[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        points_batch = self.points[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch, points_batch


# class DataProviderYoutube(DataProvider):
#     """Data provider for Youtube"""
#
#     def __init__(self, which_set='train',
#                  filepath_to_data="",
#                  batch_size=100, max_num_batches=-1,
#                  max_size=None,
#                  shuffle_order=True, rng=None):
#         """Create a new MNIST data provider object.
#
#         Args:
#             which_set: One of 'train', 'valid' or 'eval'. Determines which
#                 portion of the MNIST data this object should provide.
#             batch_size (int): Number of data points to include in each batch.
#             max_num_batches (int): Maximum number of batches to iterate over
#                 in an epoch. If `max_num_batches * batch_size > num_data` then
#                 only as many batches as the data can be split into will be
#                 used. If set to -1 all of the data will be used.
#             shuffle_order (bool): Whether to randomly permute the order of
#                 the data before each epoch.
#             rng (RandomState): A seeded random number generator.
#         """
#         # check a valid which_set was provided
#         assert which_set in ['train', 'valid', 'test'], (
#             'Expected which_set to be either train, valid or eval. '
#             'Got {0}'.format(which_set)
#         )
#         self.which_set = which_set
#         # construct path to data using os.path.join to ensure the correct path
#         # separator for the current platform / OS is used
#         # MLP_DATA_DIR environment variable should point to the data directory
#
#         data_transform = transforms.Compose([Rescale(250),
#                                              RandomCrop(224),
#                                              Normalize(),
#                                              ToTensor()])
#         assert (data_transform is not None), 'Define a data_transform'
#
#         internal_name={"train":"training","valid":"test","test":"test"}
#
#         transformed_dataset = DatasetYoutube(csv_file=os.path.join(filepath_to_data, '{0}_frames_keypoints.csv'.format(internal_name[which_set])),
#                                              root_dir=os.path.join(filepath_to_data ,internal_name[which_set]),
#                                              max_size= max_size,
#                                              transform=data_transform)
#         inputs, targets = transformed_dataset.get_data()
#         # pass the loaded data to the parent class __init__
#         super(DataProviderYoutube, self).__init__(
#             inputs, targets, batch_size, max_num_batches, shuffle_order, rng)
#
#     def __len__(self):
#         return self.num_batches
#
#     def next(self):
#         """Returns next data batch or raises `StopIteration` if at end."""
#         inputs_batch, targets_batch = super(DataProviderYoutube, self).next()
#         return inputs_batch, targets_batch
#

class DataProviderFLD(DataProvider):
    """Data provider for 300W Faces."""

    def __init__(self, dataset, which_set='train',
                 batch_size=100, max_num_batches=-1,
                 shuffle_order=False, rng=None):
        """Create a new MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.data_set=dataset
        inputs, targets, points = self.data_set.get_data(self.which_set)
        # pass the loaded data to the parent class __init__
        super(DataProviderFLD, self).__init__(
            inputs, targets, points, batch_size, max_num_batches, shuffle_order, rng)

    def __len__(self):
        return self.num_batches

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch, points_batch = super(DataProviderFLD, self).next()
        return inputs_batch, targets_batch, points_batch

    def render(self,x,y,p,out,number_images):
        self.data_set.render(x,y,p,out,number_images)


