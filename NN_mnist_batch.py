"""
Name: NN_mnist_batch.py
Description:
    NN_mnistと同様の処理をバッチにして行う．
"""

from NN_mnist import get_data, init_network, predict
import numpy as np

if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        t_batch = t[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p==t_batch)

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
