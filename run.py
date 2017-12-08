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

bandwidth = 0.5
learning_rate = 1e-1

module_name = input('Enter task location: ')
module = importlib.import_module(module_name)

N, D_in, H, D_out = 64, 1, 50, 30, 1

model = to.nn.Sequential()
for 
model.add_module(Linear(D_in, H_1))
model.add_module(Linear(D_in, H_1))
model.add_module(Linear(D_in, H_1))
            SGPA(H_1, H_2),
            to.nn.Linear(H_2, D_out)

loss_fn = to.nn.MSELoss(size_average=False)

optimizer = to.optim.Adam(model.parameters(), lr=learning_rate)

def train_step(t):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

module.load_problem(model, train_step)
