"""
Name: two_layer_net.py
Description:
    ゼロから作る Deep Learning 4章で紹介された2層のニューラルネットを実装する．
"""

import numpy as np
from activation_funcs import *
from loss_funcs import cross_entropy_error
from gradient_2d import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = sigmoid_func(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax_func(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.size != y.size:    # tがone-hot表現なら整形
            t = np.argmax(t, axis=0)
            print(t)

        acc = np.sum(y == t) / float(x.shape[0])
        return acc

    def numerical_grad(self, x, t):
        loss_W = lambda w: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
