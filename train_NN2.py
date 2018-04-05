"""
Name: train_NN2.py
Description:
    train_NN.py とほぼ同じ内容. 勾配を求めるために誤差逆伝播法を使用していることに注意
"""

"""
Name: train_NN.py
Description:
    ゼロから作る Deep Learning の4.5.2節，ミニバッチ学習
"""

import numpy as np
import matplotlib.pylab as plt
from train_NN import get_mnist_data
from two_layer_net2 import TwoLayerNet

train_loss_list = []
train_acc_list = []
test_acc_list = []

def SGD(network, x, t, batch_size, iter=6000, learning_rate=0.1):
    train_size = x.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iter):
        print(i)
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x[batch_mask]
        t_batch = t[batch_mask]
        #print(x_batch)

        #勾配の計算 (誤差逆伝播法)
        #grad = network.numerical_grad(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        #パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        #エポック毎に正解率を計算
        train_acc = network.accuracy(x, t)
        test_acc = network.accuracy(x, t)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        if i % iter_per_epoch == 0:
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            #print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    return network

def main():
    [x_train, t_train, x_test, t_test] = get_mnist_data()
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    train_net = SGD(network, x_train, t_train, batch_size=100)

    #グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
