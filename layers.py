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


class Dropout:
    """
    #参考: http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
        

class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

"""
class BatchNormalization:
"""
#参考: http://arxiv.org/abs/1502.03167
"""
def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
    self.gamma = gamma
    self.beta = beta
    self.momentum = momentum
    self.input_shape = None # Conv層の場合は4次元,全結合相の場合は2次元

    #テストで使用する平均と分散
    self.running_mean = running_mean
    self.running_var = running_var

    #backward時に使用する中間データ
    self.batch_size = None
    self.xc = None
    self.std = None
    self.dgamma = None
    self.dbeta = None

def forward(self, x, train_flg=True):
    self.input_shape = x.shape
    if x.ndim != 2:
        n, c, h, w = x.shape
        x = x.reshape(n, -1)

    out = self.__forward(x, train_flg)

    return out.reshape(*self.input_shape)


def __forward(self, x, train_flg):
    if self.running_mean is None:
        n, d = x.shape
        self.running_mean = np.zeros(d)
        self.running_var = np.zeros(d)

    if train_flg:
        mu = x.mean(axis=0)
        xc = x - mu
        var = np.mean(xc**2, axis=0)
        std = np.sqrt(var + 10e-7)
        xn = xc / std

        self.batch_size = x.shape[0]
        self.xc = xc
        self.xn = xn
        self.std = std
        self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
        self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
    else:
        xc = x - self.running_mean
        xn = xc / ((np.sqrt(self.running_var + 10e-7)))

    out = x - self.running_mean
    return out

def backward(self, dout):
    if dout.ndim != 2:
        n, c, h, w = dout.shape
        dout = dout.reshape(n, -1)

    dx = self.__backward(dout)
    dx = dx.reshape(*self.input_shape)
    return dx

def __backward(self, dout):
    dbeta = dout.sum(axis=0)
    dgamma = np.sum(self.xn * dout, axis=0)
    dxn = self.gamma * dout
    dxc = dxn / self.std
    dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
    dvar = 0.5 * dstd / self.std
    dxc += (2.0 / self.batch_size) * self.xc * dvar
    dmu = np.sum(dxc, axis=0)
    dx = dxc - dmu / self.batch_size

    self.dgamma = dgamma
    self.dbeta = dbeta

    return dx
"""


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
