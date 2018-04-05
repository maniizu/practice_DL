"""
Name: NN_mnist.py
Description:
    学習済みのネットワークを使って画像を数字ごとに分類する.
    データセットは MNIST
"""

import numpy as np
from sklearn.datasets import fetch_mldata
from activation_funcs import sigmoid_func, softmax_func
import pickle

def get_data():
    """
    MNIST データを読み込む．
    """
    mnist = fetch_mldata('MNIST original', data_home='./')
    x_test = mnist.data[60000:] / 255   #正規化
    t_test = mnist.target[60000:]
    return x_test, t_test

def init_network():
    """
    学習済みネットワークを読み込む．
    sample_weight.pklファイルがカレントディレクトリにあることを確認する．
    """
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    """
    33layer_NN.py と同様に出力を推定する．
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_func(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_func(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_func(a3)

    return y

if __name__ == '__main__':
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0

    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)    #最も確率の高いラベルを取得
        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
