"""
Name: gradient_method.py
Description:
    勾配降下法を実装する
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
from gradient_2d import numerical_gradient, function_2

def gradient_descent(f, init_x, lr=0.01, step_num=1000):
    """
    勾配降下法によって最適解に近づくための関数
    デフォルトの学習率の初期値は0.01，
    ステップ数の初期値は1000
    """
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)

if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    lr = 0.1
    step_num = 20

    x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

    x_circ = np.linspace(-5, 5, 1000)
    y_circ = np.linspace(-5, 5, 1000)
    X_circ, Y_circ = np.meshgrid(x_circ, y_circ)
    Z_circ = np.sqrt((27.5 * X_circ**2 / 9) + Y_circ**2)    # だいたい円形になる
    plt.contour(X_circ, Y_circ, Z_circ, colors=['gray'], linestyles = 'dashed')
    #plt.gca().set_aspect('equal')

    #plt.plot([-5,5], [0,0], '--b')
    #plt.plot([0,0], [-5,5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()
