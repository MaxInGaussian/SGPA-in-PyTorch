import numpy as np
import torch as to
from torch.autograd import Variable
from matplotlib import cm
from matplotlib import pylab
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import KernelDensity
from Modules import SGPA

bandwidth = 0.5

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1, 30, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(to.randn(N, D_in)*5)
y = Variable(to.randn(N, 1)+5*to.sin(x).data, requires_grad=False)

# Use the nn package to define our model and loss function.
model = to.nn.Sequential(
          to.nn.Linear(D_in, H),
          SGPA(H, H),
          to.nn.Linear(H, 1),
        )

def animate(fig):
    fig.clf()
    ax = fig.add_subplot(111)
    pts = 300
    pylab.plot(x.data.numpy().ravel(), y.data.numpy().ravel(), 'r.')
    x_plot = Variable(to.linspace(to.min(x.data), to.max(x.data), pts)[:, None])
    y_plot = to.Tensor(0, 1)
    for _ in range(100):
        y_plot = to.cat([y_plot, model(x_plot).data], 1)
    y_mu, y_std = to.mean(y_plot, 1).numpy(), to.std(y_plot, 1).numpy()
    pylab.plot(x_plot.data.numpy().ravel(), y_mu, 'k')
    errors = [0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, 2.2]
    for e in errors:
        ax.fill_between(x_plot.data.numpy().ravel(), y_mu-e*y_std, y_mu+e*y_std,
                        alpha=((3-e)/5.5)**1.7, facecolor='b', linewidth=1e-3)
    # ax.plot(Xs[:, 0], mu, alpha=0.8, c='black')
    # ax.errorbar(self.model.Xt[:, 0],
    #     self.model.yt.ravel(), fmt='r.', markersize=5, alpha=0.6)
    # ax.errorbar(self.model.Xs[:, 0],
    #     self.model.ys.ravel(), fmt='g.', markersize=5, alpha=0.6)
    # ax.set_ylim([y.min(), y.max()])
    # ax.set_xlim([-0.1, 1.1])
    pylab.pause(0.1)

def kde_animate(fig):
    fig.clf()
    ax = fig.add_subplot(111)
    pts = 300
    pylab.plot(x.data.numpy().ravel(), y.data.numpy().ravel(), 'r.')
    x_plot = Variable(to.linspace(to.min(x.data), to.max(x.data), pts)[:, None])
    y_plot = np.linspace(to.min(y.data)-.5, to.max(y.data)+.5, pts)[:, None]
    y_pred = to.Tensor(0, 1)
    for _ in range(100):
        y_pred = to.cat([y_pred, model(x_plot).data], 1)
    y_mu, y_std = to.mean(y_pred, 1).numpy(), to.std(y_pred, 1).numpy()
    # pylab.plot(x_plot.data.numpy().ravel(), y_mu, 'k')
    kde_skl = KernelDensity(bandwidth=bandwidth)
    grid = []
    for i in range(pts):
        kde_skl.fit(y_pred.numpy()[i][:, None])
        log_pdf = kde_skl.score_samples(y_plot)
        grid.append(np.exp(log_pdf)[:, None])
    grid = np.asarray(grid).reshape((pts, pts)).T
    pylab.imshow(grid, extent=(
            x_plot.data.numpy().min(), x_plot.data.numpy().max(),
            y_plot.max(), y_plot.min()),
        interpolation='bicubic', cmap=cm.Blues)
    pylab.pause(.5)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = to.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-1
optimizer = to.optim.Adam(model.parameters(), lr=learning_rate)
fig = pylab.figure()
fig.set_tight_layout(True)
pylab.show()
for t in range(100):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)
    
    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])
    
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    
    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()
    kde_animate(fig)
