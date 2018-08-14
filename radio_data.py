from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pickle as p
import platform
import matplotlib.pyplot as plt

rfpath = '/Users/ty/Desktop/RML2016/unpack/'
mod_type = ["8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
train_num = 4000
test_num = 100
class_num = 10
duration = 128


# def read_train_data(snr):
#     train_data = np.zeros([train_num*class_num, 2, duration, 1])
#     train_label = np.zeros([train_num*class_num, class_num])
#     for i in range(class_num):
#         with open('E:/DataSet/Radio/unpack/'+"%s_%s.dat" % (mod_type[i], snr), "rb") as sf:
#             samples = p.load(sf, encoding='iso-8859-1')
#             samples = samples[0:train_num, :, :].reshape([train_num, 2, duration, 1])
#             train_data[i*train_num:(i+1)*train_num, :, :, :] = samples
#             train_label[i*train_num:(i+1)*train_num, i] = np.ones(train_num)
#     np.random.seed(10)
#     np.random.shuffle(train_data)
#     np.random.seed(10)
#     np.random.shuffle(train_label)
#     return train_data, train_label


def read_train_data(snr):
    data = np.zeros([train_num*class_num, 2*duration+class_num])
    for i in range(class_num):
        with open(rfpath+"%s_%s.dat" % (mod_type[i], snr), "rb") as sf:
            samples = p.load(sf, encoding='iso-8859-1')
            samples = samples[0:train_num, :, :].reshape([train_num, 2*duration])
            data[i*train_num:(i+1)*train_num, 0:2*duration] = samples
            data[i*train_num:(i + 1) * train_num, -10+i] = np.ones(train_num)
    np.random.shuffle(data)
    train_data = data[:, 0:2*duration].reshape([train_num*class_num, 2, duration, 1])
    train_label = data[:, -10:]
    return train_data, train_label


def read_test_data(snr):
    test_data = np.zeros([test_num*class_num, 2, duration, 1])
    test_label = np.zeros([test_num*class_num, class_num])
    for i in range(class_num):
        with open(rfpath+"%s_%s.dat" % (mod_type[i], snr), "rb") as sf:
            samples = p.load(sf, encoding='iso-8859-1')
            samples = samples[-test_num:, :, :].reshape([test_num, 2, duration, 1])
            test_data[i*test_num:(i+1)*test_num, :, :, :] = samples
            test_label[i*test_num:(i+1)*test_num, i] = np.ones(test_num)
    return test_data, test_label


if __name__ == "__main__":
    test_data, test_label = read_test_data(0)
    train_data, train_label = read_train_data(0)
    print(test_data.shape, test_label.shape, train_data.shape, train_label.shape)
