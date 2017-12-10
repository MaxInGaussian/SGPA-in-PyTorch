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

import importlib
import numpy as np
import torch as to
from torch.nn import Linear
from torch.autograd import Variable

from sgpa import SGPA

tol = 1e-5
n_basis = 50
learning_rate = 1e-1

module_name = input('Enter task location: ')
module = importlib.import_module(module_name)
(N, D_in, D_out), train, eval = module.load_problem()

hidden_layers = [50]
layer_sizes = [D_in]+hidden_layers+[D_out]

model = to.nn.Sequential()
for l, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    if(l == 0):
        model.add_module('linear_l'+str(l), Linear(n_in, n_out))
        continue
    model.add_module('sgpa_l'+str(l), SGPA(n_in, n_basis))
    model.add_module('linear_l'+str(l), Linear(n_basis, n_out))

loss_fn = to.nn.MSELoss(size_average=False)

optimizer = to.optim.Adam(model.parameters(), lr=learning_rate)
last_t, min_loss, best_state = 0, np.Infinity, None

def train_op(t, x, y, x_val=None, y_val=None, locals=None):
    loss = loss_fn(model(x), y)
    if(x_val is None):
        if(t > locals[last_t]):
            print('Epoch %d: Train Loss = %.5f'%(t, loss))
    else:
        loss_val = loss_fn(model(x_val), y_val)
        if(loss_val < min_loss+tol):
            min_loss = loss_val
            best_state = model.state_dict()
        print('=> Valid Loss = %.5f (Best = %.5f)'%(loss_val, min_loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    last_t = t

train(model, train_op, locals())
model.load_state_dict(best_state)