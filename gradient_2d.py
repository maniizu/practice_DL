"""
Name: gradient_2d.py
Description:
    ゼロから作る Deep Learning の4.4節で導入された勾配法のプログラム
"""

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def _numerical_gradient_no_batch(f, x):
    """
    数値微分によって勾配を求める関数
    ただし，バッチ処理はできない
    """
    h = 1e-4
    grad = np.zeros_like(x)

    for index in range(x.size):
        tmp = x[index]

        # f(x + h)を求める
        x[index] = tmp + h
        fxh1 = f(x)

        # f(x - h)を求める
        x[index] = tmp - h
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2 * h)

        x[index] = tmp  # もとに戻す

    return grad

def numerical_gradient(f, x):
    """
    数値微分によって勾配を求める関数
    バッチ処理にも対応
    """
    if x.ndim == 1:
        return _numerical_gradient_no_batch(f, x)
    else:
        grad = np.zeros_like(x)

        for index, i in enumerate(x):
            grad[index] = _numerical_gradient_no_batch(f, i)

        return grad


def function_2(x):
    """
    例としてとりあえず作る関数
    """
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles='xy', color='#666666')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    #plt.legend()
    plt.draw()
    plt.show()
