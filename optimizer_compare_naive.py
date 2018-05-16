"""
Name: optimizer_compare_naive.py
Usage: ターミナルで python3 optimizer_compare_naive.py
Description:
    optimizers.py で実装した最適化手法を比較するプログラム
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimizers import *


def f(x, y):
    return x**2 / 20.0 + y**2

def df(x, y):
    return x / 10.0, 2.0 * y

if __name__ == '__main__':
    # 初期値を設定
    init_pos = (-7.0, 2.0)
    params = {}
    params['x'], params['y'] = init_pos[0], init_pos[1]
    grads = {}
    grads['x'], grads['y'] = 0, 0

    # 最適化手法を設定
    opts = OrderedDict()
    opts['SGD'] = SGD(lr=0.95)
    opts['Momentum'] = Momentum(lr=0.1)
    opts['AdaGrad'] = AdaGrad(lr=1.5)
    opts['RMSprop'] = RMSprop(lr=0.2)
    opts['Adam'] = Adam(lr=0.3)

    idx = 1

    for key in opts:
        opt = opts[key]
        x_history = []
        y_history = []
        params['x'], params['y'] = init_pos[0], init_pos[1]

        # パラメータを更新していく
        for i in range(30):
            x_history.append(params['x'])
            y_history.append(params['y'])

            grads['x'], grads['y'] = df(params['x'], params['y'])
            opt.update(params, grads)

        # プロット用の設定
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        mask = Z > 7
        Z[mask] = 0

        # プロット
        plt.subplot(2, 3, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-', color='red')
        plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        plt.title(key)
        plt.xlabel('x')
        plt.ylabel('y')

    plt.show()
