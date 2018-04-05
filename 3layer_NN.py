"""
Name: 3layer_NN.py
Description:
    ゼロから作る Deep Learning の3.4節で示された3層NNの実装
"""

import numpy as np
from activation_funcs import sigmoid_func, identity_func

def init_network():
    """
    ネットワークの重みを初期化
    """
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    """
    入力データxからネットワークの出力yを得る
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_func(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_func(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_func(a3)
    return y

if __name__ == '__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
