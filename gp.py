#!/usr/bin/env python3

__author__ = 'Michael Ciccotosto-Camp'
__version__ = ''

import os
import sys

import numpy as np
import scipy as sp
import pandas as pd
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
    K_ = exact_kernel(x_pred, sigma=sigma)
    Vf = 1 - np.sum(Lk**2, axis=0)
    return Ef, Vf


def sin_eg():

    x_train = np.linspace(0, 2 * np.pi, 50)
    x_test = np.linspace(0, 2 * np.pi, 250)
    x_train = np.delete(x_train, slice(3, 7))
    x_train = np.delete(x_train, slice(12, 27))
    x_train = np.delete(x_train, slice(35, 27))
    f_train = np.sin(x_train)
    y_train = f_train + np.random.randn(len(x_train)) * 0.125
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(9, 6)
    plt.style.use("fast")
    sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.scatter(x_train, y_train, s=10.0, c="k", label="data")
    plt.plot(x_test, np.sin(x_test), linewidth=2.1,
             color=r"#FF0000", label="f")
    Ef, Vf = gp_reg_pred(x_train.reshape(-1, 1), y_train,
                         x_test.reshape(-1, 1), sigma=1.1)
    stdf = np.sqrt(Vf)
    plt.plot(x_test, Ef, linewidth=2.1, color=r"#0000FF", label="E[f_*]")
    plt.gca().fill_between(x_test.flat, Ef-3*stdf,
                           Ef+3*stdf, color="#0000FF", alpha=0.1, label="S[f_*]")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), "img", "sin_eg.png"))
    return


def stocks_eg():

    df = pd.read_csv(os.path.join(os.getcwd(), "data", "stock_market.csv"))

    date_format = r"%Y-%m-%d"
    times = df["Date"]
    times = pd.to_datetime(times, format=date_format)
    times = times.diff(1).dt.days
    times = times.fillna(0.0)
    times = times.astype(float)
    df["Date"] = times.cumsum(axis="index", skipna=True).to_numpy(
        dtype=float).reshape(-1, 1)
    close = df["Close"].to_numpy(dtype=float)
    time = df["Date"].to_numpy(dtype=float)

    time_all = time[3800:4500]
    close_all = close[3800:4500] / 1000

    time_all = time_all - min(time_all)

    time_train = time_all[:-200]
    close_train = close_all[:-200]
    time_test = time_all[-200:]
    close_test = close_all[-200:]

    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(9, 6)
    plt.style.use("fast")
    sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    plt.scatter(time_train, close_train, s=10.0, c="k", label="data")
    plt.scatter(time_test, close_test, s=80,
                c="k", label="actual", marker="+")
    Ef, Vf = gp_reg_pred(time_train.reshape(-1, 1), close_train.squeeze(),
                         time_all.reshape(-1, 1), sigma=900)
    stdf = np.sqrt(Vf)
    plt.plot(time_all, Ef, linewidth=2.1, color=r"#0000FF", label="E[f_*]")
    plt.gca().fill_between(time_all.flat, Ef-3*stdf,
                           Ef+3*stdf, color="#0000FF", alpha=0.1, label="S[f_*]")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), "img", "stock_eg.png"))
    return


def main():

    # sin_eg()
    stocks_eg()

    return


if __name__ == "__main__":
    main()
