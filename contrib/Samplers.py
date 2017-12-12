# MIT License
# 
# Copyright (c) 2017 Max W. Y. Lam
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

import torch
from torch.utils.data.sampler import RandomSampler


class TrainSampler(RandomSampler):
    """
    Samples elements randomly given train porportion, without replacement.

    Arguments:
        data_source (Dataset): the full dataset to sample from
        train_proportion (float): the porportion of dataset for training
    """

    def __init__(self, data_source, train_proportion=0.6):
        super(TrainSampler, self).__init__(data_source)
        self.train_proportion = train_proportion
        self.train_size = int(len(data_source)*self.train_proportion)
        self.train_indices = torch.randperm(len(data_source))[:self.train_size]

    def __iter__(self):
        return iter(self.train_indices)

    def __len__(self):
        return self.train_size


class ValidationSampler(RandomSampler):
    """
    Samples elements randomly given validation porportion, without replacement.

    Arguments:
        train_sampler (TrainSampler): the sampler for training
        valid_proportion (float): the porportion of dataset for validation
    """

    def __init__(self, train_sampler, valid_proportion=0.2):
        super(ValidationSampler, self).__init__(train_sampler.data_source)
        self.train_indices = train_sampler.train_indices.numpy()
        self.valid_proportion = valid_proportion
        self.valid_size = int(len(self.data_source)*self.valid_proportion)
        self.valid_indices = np.setdiff1d(range(len(self.data_source)),
            self.train_indices)[:self.valid_size]

    def __iter__(self):
        return iter(self.valid_indices)

    def __len__(self):
        return self.valid_size


class TestSampler(RandomSampler):
    """
    Samples elements randomly given test porportion, without replacement.

    Arguments:
        valid_sampler (ValidationSampler): the sampler for validation
        test_proportion (float): the porportion of dataset for testing
    """

    def __init__(self, valid_sampler, test_proportion=0.2):
        super(TestSampler, self).__init__(valid_sampler.data_source)
        self.train_indices = valid_sampler.train_indices
        self.valid_indices = valid_sampler.valid_indices
        self.test_proportion = test_proportion
        self.test_size = int(len(self.data_source)*self.test_proportion)
        self.test_indices = np.setdiff1d(range(len(self.data_source)),np.hstack(
            (self.train_indices, self.valid_indices)))[:self.test_size]

    def __iter__(self):
        train_size = self.__len__()
        return iter(self.test_indices)

    def __len__(self):
        return self.test_size