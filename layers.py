"""
Name: layers.py
Description:
    ゼロから作る Deep Learning で実装されたレイヤをまとめておく
    ターミナルから直接実行するときは, 書籍中の buy_apple.py と
    buy_apple_orange.py と同様の動作をする.
"""

import numpy as np
from activation_funcs import softmax_func, sigmoid_func
from loss_funcs import cross_entropy_error

class MulLayer:
    """
    乗算レイヤ
    """
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer:
    """
    加算レイヤ
    """
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy

class Relu:
    """
    Rectified Linear Unit レイヤ
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    """
    Sigmoid レイヤ
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid_func(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class AffineMat:
    """
    行列のみを考慮したアフィンレイヤ
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class Affine:
    """
    テンソル対応したアフィンレイヤ
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape) #入力データの形状に戻す (テンソル対応)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax_func(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # tがone-hot表現になっている場合
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        # それ以外
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


if __name__ == '__main__':
    # まずは buy_apple.py を動作させる.
    # 初期値
    apple = 100
    apple_num = 2
    tax = 1.1

    # レイヤ
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # 順伝播
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)
    print('リンゴ2個の値段は' + str(int(price)) + '円です')

    #逆伝播
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print('リンゴ, リンゴの個数, 税金の各微分値: ' + '{:.1f}'.format(dapple) +\
        ', ' + str(int(dapple_num)) + ', ' + str(int(dtax)))


    # 次に buy_apple_orange.py を動作させる.
    # 追加の初期値
    orange = 150
    orange_num = 3

    # レイヤ (乗算レイヤは前の代入した値が保存されている可能性があるため,再度初期化)
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # 順伝播
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)
    print('リンゴ2個とみかん3個の値段は' + str(int(price)) + '円です')

    # 逆伝播
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print('リンゴ, リンゴの個数, みかん, みかんの個数, 税金の各微分値: ' +\
        '{:.1f}'.format(dapple) + ', ' + str(int(dapple_num)) + ', ' +\
            '{:.1f}'.format(dorange) + ', ' + str(int(dorange_num)) + ', ' + str(dtax))
