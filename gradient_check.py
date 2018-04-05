"""
Name: gradient_check.py
Description:
    数値微分による勾配と誤差逆伝播法による勾配が(ほとんど)一致することを確認する
"""

import numpy as np
from train_NN import get_mnist_data
from two_layer_net2 import TwoLayerNet

if __name__ == '__main__':
    #データの読み込み
    [x_train, t_train, x_test, t_test] = get_mnist_data()

    #ネットワークを初期化
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    #テスト用のデータ
    x_batch = x_train[:3]
    t_batch = t_train[:3]

    #各手法で勾配を計算
    grad_numerical = network.numerical_grad(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    #各重みの絶対誤差を求める
    for key in grad_numerical.keys():
        diff = np.average(grad_backprop[key] - grad_numerical[key])
        print(key + ':' + str(diff))
