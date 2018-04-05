"""
Name: gradient_simplenet.py
Description:
    ゼロから作る Deep Learning 4.4.2節で紹介された NN の勾配法
"""

import numpy as np
from activation_funcs import softmax_func
from loss_funcs import cross_entropy_error
from gradient_2d import numerical_gradient

class simpleNet:
    def __init__(self):
         self.W = np.random.randn(2,3)  #ガウス分布でパラメータを初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax_func(z)
        loss = cross_entropy_error(y, t)
        return loss

if __name__ == '__main__':
    net = simpleNet()
    print('パラメータの初期値')
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print('初期パラメータを使用した推論')
    print(p)
    print('推論結果: ' + str(np.argmax(p)))
    t = np.array([0, 0, 1])
    print('損失関数の値(ラベル): ' + str(net.loss(x, t)))

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)

    print('更新後のパラメータ')
    print(dW)
