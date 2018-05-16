"""
Name: util.py
Usage: 他のファイルから関数を呼び出す
Description:
    ゼロから作る Deep Learning で使用されている便利な関数をまとめておく
"""

import numpy as np


def smooth_curve(x):
    """
    損失関数のグラフをなめらかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


#def smooth_curve(x):
"""損失関数のグラフを滑らかにするために用いる
参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
"""
"""
window_len = 11
s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
w = np.kaiser(window_len, 2)
y = np.convolve(w/w.sum(), s, mode='valid')
return y[5:len(y)-5]
"""


def shuffle_dataset(x, t):
    """
    データセットをシャッフルする

    params:
        x: 訓練データ
        t: 教師ラベル

    returns:
        x: シャッフルした訓練データ
        t: シャッフルした訓練データに対応する教師ラベル
    """
    permutation = np.random.permutation(x.shape[0])
    if x.ndim == 2:
        x = x[permutation,:]
    else:
        x[permutation,:,:,:]
    t = t[permutation]

    return x, t
