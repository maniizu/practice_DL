"""
Name: overfit_weight_decay.py
Usage: ターミナルで python3 overfit_weight_decay.py と入力
Description:
    ゼロから始まる Deep Learning の6.4.2章の比較実験
"""

import numpy as np
import matplotlib.pyplot as plt
from train_NN import get_mnist_data
from multi_layer_net import MultiLayerNet
from optimizers import SGD
from util import shuffle_dataset


def main():
    [x_train, t_train, x_test, t_test] = get_mnist_data()

    # データをシャッフル
    x_train, t_train = shuffle_dataset(x_train, t_train)

    #過学習を再現するために，学習データを削減
    x_train = x_train[:300]
    t_train = t_train[:300]

    # weight decay の設定
    weight_decay_lambda = 0.1

    #ネットワークなどの設定
    network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100], \
        output_size=10, weight_decay_lambda=weight_decay_lambda)

    opt = SGD(lr=0.01)

    max_epochs = 201
    train_size = x_train.shape[0]
    batch_size = 100

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(100000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = network.gradient(x_batch, t_batch)
        opt.update(network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print("epoch: " + str(epoch_cnt) + ", train acc: " + str(train_acc) \
                + ", test acc: " + str(test_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    # グラフの描画
    #markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.xlim(0, max_epochs)
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
