"""
Name: gradient_1d.py
Description:
    ゼロから作る Deep Learning の4.3.2節にある数値微分の例
"""

import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    """
    数値微分をする関数．中心差分を使用していることに注意する．
    """
    h = 1e-4    # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    """
    例として作った適当な関数
    """
    return 0.01 * x ** 2 + 0.1 * x

def tangent_line(f, x):
    """
    接線を表す関数を返す
    """
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d*t + y

if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)   # 0から20まで，0.1刻みの配列
    y = function_1(x)
    plt.xlabel('x')
    plt.ylabel('f(x)')

    tf = tangent_line(function_1, 5)
    y2 = tf(x)

    tf2 = tangent_line(function_1, 10)
    y3 = tf2(x)

    plt.plot(x, y, label='original func')
    plt.plot(x, y2, label='differentiation at 5')
    plt.plot(x, y3, label='differentiation at 10')
    plt.legend()
    plt.show()
