"""
Name: optimizer_compare_mnist.py
Usage: ターミナルで python3 optimizer_compare_mnist と入力
Description:
    ゼロから作る Deep Learning の6.1.8節のコード
"""

import numpy as np
import matplotlib.pyplot as plt
from train_NN import get_mnist_data
from optimizers import *
from multi_layer_net import MultiLayerNet
from util import smooth_curve


def main():
    # MNISTデータの読み込み
    [x_train, t_train, x_test, t_test] = get_mnist_data()

    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000


    # 実験の設定
    opts = {}
    opts['SGD'] = SGD()
    opts['Momentum'] = Momentum()
    opts['AdaGrad'] = AdaGrad()
    opts['Adam'] = Adam()
    opts['RMSprop'] = RMSprop()

    networks = {}
    train_loss = {}
    for key in opts.keys():
        networks[key] = MultiLayerNet(input_size=784,\
            hidden_size_list = [100, 100, 100, 100], output_size = 10)
        train_loss[key] = []


    # 訓練開始
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in opts.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            opts[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        if (i+1) % 100 == 0:
            print("============" + "iteration:" + str(i+1) + "=============")
            for key in opts.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))


    # グラフの描画
    markers = {'SGD': 'o', 'Momentum': 'x', 'AdaGrad': 's', 'Adam': 'D', 'RMSprop': '^'}
    x = np.arange(max_iterations)
    for key in opts.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], \
            markevery=100, label=key)

    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
