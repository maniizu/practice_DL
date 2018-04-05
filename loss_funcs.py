"""
Name: loss_funcs.py
Description:
    ゼロから作る Deep Learning の4.2節で紹介された損失関数の実装
"""

import numpy as np

def mean_squared_error(y, t):
    """
    二乗和誤差
    """
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    """
    交差エントロピー誤差 (ミニバッチ対応済み)
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    #教師データが one-hot表現の場合，正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

"""
def cross_entropy_error(y, t):
"""
    #交差エントロピー誤差 (ミニバッチ対応済み)
"""
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)

    batch_size = y.shape[0]
    delta = 1e-7    # マイナス無限大防止用

    # t が one-hot 表現の場合
    if t.size == y.size:
        return -np.sum(t * np.log(y + delta)) / batch_size
    # t がラベル表現の場合
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
"""

if __name__ == '__main__':
    # 二乗和誤差の確認
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(mean_squared_error(y, t))
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0])
    print(mean_squared_error(y, t))

    # 交差エントロピー誤差の確認
    # one-hot 表現
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(cross_entropy_error(y, t))
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0])
    print(cross_entropy_error(y, t))

    # ラベル表現
    t = np.array([2])
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(cross_entropy_error(y, t))
    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0])
    print(cross_entropy_error(y, t))

    t = np.array([2, 3])
    y = np.array([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
                [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]])
    print(type(t))
    print(cross_entropy_error(y, t))
