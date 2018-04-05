"""
Name: activation_funcs.py
Description:
    3.2節で紹介された活性化関数をまとめておく．
"""

import numpy as np
import matplotlib.pylab as plt

def step_func(x):
    """
    ステップ関数
    """
    return np.array(x>0, dtype=np.int)

def sigmoid_func(x):
    """
    シグモイド関数
    """
    return 1/(1 + np.exp(-x))

def relu(x):
    """
    Rectified Linear Unit (ReLU) 関数
    """
    return np.maximum(0, x)

def identity_func(x):
    """
    恒等関数
    """
    return x

def softmax_func(x):
    """
    ソフトマックス関数
    ただし，xの各要素をxの最大値で引いて，オーバーフローを抑える
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

if __name__ == '__main__':
    x = np.arange(-4.0, 4.0, 0.1)
    y_step = step_func(x)
    y_sigmoid = sigmoid_func(x)
    y_relu = relu(x)
    plt.plot(x, y_step)
    plt.plot(x, y_sigmoid)
    plt.plot(x, y_relu)
    plt.ylim(-0.1, 4.1)
    plt.show()
