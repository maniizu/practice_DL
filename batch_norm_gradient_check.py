"""
Name: batch_norm_gradient_check.py
Usage: ターミナルで python3 batch_norm_gradient_check.py
Description:
    Batch Normalization の更新がうまくいくか調べる．
"""

import numpy as np
from train_NN import get_mnist_data
from multi_layer_net_extend import MultiLayerNetExtend

#データの読み込み
[x_train, t_train, x_test, t_test] = get_mnist_data()

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100], \
    output_size=10, use_batchnorm=True)

x_batch = x_train[:1]
t_batch = t_train[:1]

grad_backprop = network.gradient(x_batch, t_batch)
grad_numerical = network.numerical_gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
