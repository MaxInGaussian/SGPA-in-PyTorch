''' load_data.py '''
import numpy as np
import pandas as pd

DATA_PATH = 'housing.data'

## Load and return K sets of data (each contains train, validation, and test)
def load_data(n_folds):
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

## Standardize the scale of input-output pairs to N(0, 1)
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
    
    x, y = load_data(5)
    
    N, D_in, D_out = 64, 1, 1
    
    def run(model, train_step):
        x = Variable(to.randn(N, 1)*3)
        y = Variable(to.randn(N, 1)*.5+5*to.sin(x).data, requires_grad=False)
        
        fig = pylab.figure()
        fig.set_tight_layout(True)
        
        def ani(t):
            train_step(t, x, y)
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
    
    return (N, D_in, D_out), run