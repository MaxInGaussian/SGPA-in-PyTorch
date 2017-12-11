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
import torch as to
from torch.autograd import Variable

DATA_PATH = 'uci_reg/housing.data'

def load_data(n_folds):
    import pandas as pd
    np.random.seed(314159)
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=None, index_col=None, sep="[ ^]+")
    data = data.sample(frac=1).dropna(axis=0).as_matrix().astype(np.float32)
    X, Y = data[:, :-1], data[:, -1]
    Y = Y[:, None]
    n_data = Y.shape[0]
    n_partition = n_data//n_folds
    n_train = n_partition*(n_folds-1)
    dataset, folds = [], []
    for i in range(n_folds):
        if(i == n_folds-1):
            fold_inds = np.arange(n_data)[i*n_partition:]
        else:
            fold_inds = np.arange(n_data)[i*n_partition:(i+1)*n_partition]
        folds.append([X[fold_inds], Y[fold_inds]])
    for i in range(n_folds):
        valid_fold, test_fold = i, (i+1)%n_folds
        train_folds = np.setdiff1d(np.arange(n_folds), [test_fold, valid_fold])
        X_train = np.vstack([folds[fold][0] for fold in train_folds])
        Y_train = np.vstack([folds[fold][1] for fold in train_folds])
        X_valid, Y_valid = folds[valid_fold]
        X_test, Y_test = folds[test_fold]
        dataset.append([X_train, Y_train, X_valid, Y_valid, X_test, Y_test])
    return dataset

def scaler(data_train, data_valid, data_test):
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    train_standardized = (data_train - mean)/std
    valid_standardized = (data_test - mean)/std
    test_standardized = (data_test - mean)/std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return train_standardized, valid_standardized, test_standardized, mean, std
    

def load_problem():
    
    dataset = load_data(5)[np.random.choice(np.arange(5))]
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = dataset
    (N, D_in), D_out = X_train.shape, Y_test.shape[-1]
    X_train, X_valid, X_test, X_mean, X_std = scaler(X_train, X_valid, X_test)
    Y_train, Y_valid, Y_test, Y_mean, Y_std = scaler(Y_train, Y_valid, Y_test)
    x_train = Variable(to.Tensor(X_train), requires_grad=False)
    y_train = Variable(to.Tensor(Y_train), requires_grad=False)
    x_valid = Variable(to.Tensor(X_valid), requires_grad=False)
    y_valid = Variable(to.Tensor(Y_valid), requires_grad=False)
    x_test = Variable(to.Tensor(X_test), requires_grad=False)
    y_test = Variable(to.Tensor(Y_test), requires_grad=False)
    
    def train(locals, model, train_op, num_epochs=1000, batch_size=20):
        for ep in range(num_epochs):
            for it in range(N//batch_size):
                X_batch = X_train[it*batch_size:(it+1)*batch_size]
                Y_batch = Y_train[it*batch_size:(it+1)*batch_size]
                x = Variable(to.Tensor(X_batch), requires_grad=False)
                y = Variable(to.Tensor(Y_batch), requires_grad=False)
                train_op(locals, ep, x, y)
            train_op(locals, ep, x_train, y_train, x_val=x_valid, y_val=y_valid)
    
    def eval(model, eval_op):
        eval_op(x_test, y_test, {'task': 'reg', 'data': [Y_mean, Y_std]})
    
    return (N, D_in, D_out), train, eval
    
    