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
from matplotlib import cm
from matplotlib import pylab
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import KernelDensity


def load_problem():
    
    N, D_in, D_out = 64, 1, 1
    
    def train(model, train_op):
        x = Variable(to.randn(N, 1)*3)
        y = Variable(to.randn(N, 1)*.5+5*to.sin(x).data, requires_grad=False)
        
        fig = pylab.figure()
        fig.set_tight_layout(True)
        
        def ani(t):
            train_op(t, x, y)
            fig.clf()
            ax = fig.add_subplot(111)
            pts = 300
            pylab.plot(x.data.numpy().ravel(), y.data.numpy().ravel(), 'r.')
            x_plot = Variable(to.linspace(to.min(x.data)-1.,
                to.max(x.data)+1., pts)[:, None])
            y_plot = np.linspace(to.min(y.data)-1.,
                to.max(y.data)+1., pts)[:, None]
            y_pred = to.Tensor(0, 1)
            for _ in range(100):
                y_pred = to.cat([y_pred, model(x_plot).data], 1)
            y_mu, y_std = to.mean(y_pred, 1).numpy(), to.std(y_pred, 1).numpy()
            kde_skl = KernelDensity(bandwidth=0.5)
            grid = []
            for i in range(pts):
                kde_skl.fit(y_pred.numpy()[i][:, None])
                log_pdf = kde_skl.score_samples(y_plot)
                grid.append(np.exp(log_pdf)[:, None])
            grid = np.asarray(grid).reshape((pts, pts)).T
            ax.imshow(grid, extent=(
                    x_plot.data.numpy().min(), x_plot.data.numpy().max(),
                    y_plot.max(), y_plot.min()),
                interpolation='bicubic', cmap=cm.Blues)
            ax.set_ylim([to.min(y.data)-1, to.max(y.data)+1])
            ax.set_xlim([to.min(x.data)-1, to.max(x.data)+1])
            pylab.pause(.5)
            return ax
    
        anim = FuncAnimation(fig, ani, frames=np.arange(0, 200), interval=300)
        anim.save('demo_1d_reg/demo_1d_reg_kde.gif', writer='imagemagick')
    
    return (N, D_in, D_out), train, None