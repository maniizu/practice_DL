"""
Name: overfit_dropout.py
Usage: ターミナルで python3 overfit_dropout.py と入力
Description:
    ゼロから作る Deep Learning の6.4.3節の比較実験
"""

import numpy as np
import matplotlib.pyplot as plt
from train_NN import get_mnist_data
from multi_layer_net_extend import MultiLayerNetExtend
from trainer import Trainer
from util import shuffle_dataset


def main():

    [x_train, t_train, x_test, t_test] = get_mnist_data()

    x_train, t_train = shuffle_dataset(x_train, t_train)

    #過学習を再現するために，学習データを削減
    x_train = x_train[:300]
    t_train = t_train[:300]

    # Dropout の有無，割合の設定
    use_dropout = True  # Dropout あり
    dropout_ratio = 0.2

    #ネットワークを生成して学習
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100,100,100], \
        output_size=10, use_dropout=use_dropout, dropout_ratio=dropout_ratio)
    trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=301, \
        mini_batch_size=100, opt='sgd', opt_param={'lr': 0.01}, verbose=True)

    trainer.train()

    train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

    #グラフの描画
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
