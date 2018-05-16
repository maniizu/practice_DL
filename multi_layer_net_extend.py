"""
Name: multi_layer_net_extend.py
Usage: 他のファイルで呼び出して使用する
Description:
    全結合の多層ニューラルネットワーク（拡張版）
"""

import numpy as np
from collections import OrderedDict
from layers import *
from multi_layer_net import numerical_gradient as numeric_grad


class MultiLayerNetExtend:
    """
    全結合の多層ニューラルネットワーク（拡張版）
    Weight Decay, Dropout, Batch Normalization の機能を持つ

    params:
        input_size: 入力サイズ(MNISTデータでは784)
        hidden_size_list: 隠れ層のニューロンの数のリスト(e.g. [100, 100, 100])
        output_size: 出力サイズ(MNISTデータでは10)
        activation: 活性化関数の指定．'relu'か'sigmoid'．初期値は'relu'
        weight_init_std: 重みの標準偏差を指定(e.g. 0.01)
            'relu'または'he'を指定した場合はHeの初期値を指定
            'sigmoid'または'xavier'を指定した場合はXavierの初期値を指定
        weight_decay_lambda: Weight Decay (L2ノルム)の強さ
        use_dropout: Dropoutを使用するかどうか
        dropout_ratio: Dropoutの割合
        use_batchnorm: Batch Normalizationを使用するかどうか
    """

    def __init__(self, input_size, hidden_size_list, output_size, \
        activation='relu', weight_init_std='relu', weight_decay_lambda=0,\
            use_dropout=False, dropout_ratio=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        #重みの初期化
        self.__init_weight(weight_init_std)

        #レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)] = \
                Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])
            if self.use_batchnorm:
                self.params['gamma'+str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta'+str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm'+str(idx)] = \
                    BatchNormalization(self.params['gamma'+str(idx)], self.params['beta'+str(idx)])

            self.layers['Activation_func'+str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout'+str(idx)] = Dropout(dropout_ratio)

        idx = self.hidden_layer_num + 1
        self.layers['Affine'+str(idx)] = Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """
        重みの初期値を設定

        params:
            weight_init_std: 重みの標準偏差を指定 (e.g. 0.01)
                'relu'または'he'を指定した場合はHeの初期値を指定
                'sigmoid'または'xavier'を指定した場合はXavierの初期値を指定
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx-1])     #ReLUを使用するときに推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx-1])     #sigmoidを使うとき
            self.params['W'+str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """
        損失関数を求める

        params:
            x: 入力データ
            t: 教師ラベル
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num+2):
            W = self.params['W'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        勾配を求める（数値微分）

        params:
            x: 入力データ
            t: 教師ラベル

        returns:
            各層の勾配を持ったディクショナリ変数
            grads['W1']，grads['W2']，...: 各層の重み
            grads['b1']，grads['b2']，...: 各層のバイアス
        """
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W'+str(idx)] = numeric_grad(loss_W, self.params['W'+str(idx)])
            grads['b'+str(idx)] = numeric_grad(loss_W, self.params['b'+str(idx)])

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma'+str(idx)] = numeric_grad(loss_W, self.params['gamma'+str(idx)])
                grads['beta'+str(idx)] = numeric_grad(loss_W, self.params['beta'+str(idx)])
        return grads


    def gradient(self, x, t):
        #forward
        self.loss(x, t, train_flg=True)

        #backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W'+str(idx)] = self.layers['Affine'+str(idx)].dW +\
                self.weight_decay_lambda * self.params['W'+str(idx)]
            grads['b'+str(idx)] = self.layers['Affine'+str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma'+str(idx)] = self.layers['BatchNorm'+str(idx)].dgamma
                grads['beta'+str(idx)] = self.layers['BatchNorm'+str(idx)].dbeta

        return grads
