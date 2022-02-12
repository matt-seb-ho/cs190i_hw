import numpy as np
from lrc import *

train_data = "Xtrain.csv"
train_labels = "Ytrain.csv"

# read data
train_data = np.genfromtxt(train_data, dtype='i1', delimiter=',')
train_labels = np.genfromtxt(train_labels, dtype='i1', delimiter=',')
# test_data = np.genfromtxt(test_data, dtype='i1', delimiter=',')

# your algorithm
# lrc = LRC(1e-5, converge_diff=1e-4)
# lrc.train(train_data, train_labels)

print('loaded train_data, train_labels')
print('LRC ctor signature:'
      'def __init__(self, learn_rate=2e-2, l2_penalty=1e-4,'
                   'converge_diff=1e-5, adagrad_eps=1e-6):')

