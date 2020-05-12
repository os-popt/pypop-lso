import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_contour(func, x, y, levels=None):
    x, y = np.array(x), np.array(y)
    if (x.ndim != 1) or (y.ndim != 1):
        raise TypeError("Both x and y should be a vector.")

    [X, Y] = np.meshgrid(x, y)
    Z = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    if levels is None:
        plt.contourf(X, Y, Z, cmap=plt.cm.cool)
        plt.contour(X, Y, Z, colors='white')
    else:
        plt.contourf(X, Y, Z, levels, cmap=plt.cm.cool)
        cs = plt.contour(X, Y, Z, levels, colors='white')
        plt.clabel(cs, inline=True, fontsize=12, colors='white')

    plt.title(func.__name__)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(func.__name__ + ".png")
    plt.show()


def plot_surface(func, x, y):
    x, y = np.array(x), np.array(y)
    if (x.ndim != 1) or (y.ndim != 1):
        raise TypeError("Both x and y should be a vector.")

    [X, Y] = np.meshgrid(x, y)
    Z = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    figure = plt.figure()
    ax = Axes3D(figure)
    ax.plot_surface(X, Y, Z, cmap=plt.cm.cool, alpha=0.9, edgecolors=None)

    plt.title(func.__name__)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(func.__name__ + ".png")
    plt.show()
