#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

from cProfile import label
import os
import sys

import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def exact_kernel(X: np.ndarray, Y: np.ndarray = None, sigma: float = 1) -> np.ndarray:
    """
    Produces the exact kernel of a matrix using a RBF kernel. If only the first
    matrix is provided then the following Gram matrix is computed:

    K_{i,j} = k(x_i, x_j)

    where x_i represents the ith row. If a value for the Y matrix is given 
    then the following Gram matrix is computed:

    K_{i,j} = k(x_i, y_j).

    The value of sigma is the variance value required for the RBF kernel.
    """
    Y = X if Y is None else Y
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-pairwise_sq_dists / (sigma**2))


def gp_reg_pred(X_train: np.ndarray, Y_train: np.ndarray, x_pred: np.ndarray,
                sigma: float = None):
    """
    Makes real-values predictions using Gaussian processes.

    Parameters:
        X_train:
            An n-by-d np.ndarray of training inputs.
        Y_train:
            A n-by-1 np.ndarray of training labels corresponding to the 
            training inputs.
        x_pred:
            An 1-by-d np.ndarray of input to make a prediction at.
        sigma:
            The bandwidth parameters of the kernel matrix. Must be 
            greater than 0.
    Returns:
        Returns the prediction as a float.
    """
    n, d = X_train.shape
    # Create the Gram matrix corresponding to the training data set.
    K = exact_kernel(X_train, sigma=sigma)
    # Noise variance of labels.
    s = np.var(Y_train.squeeze())
    L = np.linalg.cholesky(K + s*np.eye(n))
    # compute the mean at our test points.
    Lk = np.linalg.solve(L, exact_kernel(X_train, x_pred, sigma=sigma))
    Ef = np.dot(Lk.T, np.linalg.solve(L, Y_train))
    # compute the variance at our test points.
    K_ = exact_kernel(x_pred, x_pred, sigma=sigma)
    Vf = np.diag(K_) - np.sum(Lk**2, axis=0)
    return Ef, Vf


def sin_eg():

    x_train = np.linspace(0, 2 * np.pi, 50)
    x_test = np.linspace(0, 2 * np.pi, 250)
    x_train = np.delete(x_train, slice(3, 7))
    x_train = np.delete(x_train, slice(20, 25))
    x_train = np.delete(x_train, slice(35, 27))
    f_train = np.sin(x_train)
    y_train = f_train + np.random.rand(len(x_train)) / 5

    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(9, 6)
    plt.style.use("fast")
    sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.scatter(x_train, y_train, s=7.0, c="k", label="data")
    Ef, Vf = gp_reg_pred(x_train.reshape(-1, 1), y_train,
                         x_test.reshape(-1, 1), sigma=1.0)
    plt.plot(x_test, Ef, c=r"#0000FF", label="E[f_*]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(x_train)


def main():

    sin_eg()

    return


if __name__ == "__main__":
    main()
