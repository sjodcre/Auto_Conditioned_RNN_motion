import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from Network import Network
from AdamTrainer import AdamTrainer
from network import create_core, create_regressor

rng = np.random.RandomState(23455)
#hmd05 is original
data = np.load('../data/processed/data_hdm05.npz')
#data = np.load('../data/processed/data_cmu.npz')
data_kicking = np.hstack([np.arange(199, 246), np.arange(862, 906), np.arange(1582,1640), np.arange(2188,2233), np.arange(2796,2844)])
rng.shuffle(data_kicking)

kicking_train = data_kicking[:len(data_kicking)//2]
kicking_valid = data_kicking[len(data_kicking)//2:]
#data_kicking_cmu=np.hstack([1542,1543,1545,1546,1548,1553,1554,1555,1556,2278,2279])
data_kicking_cmu=np.hstack([np.arange(1541, 1545),np.arange(1545,1548),np.arange(1548, 1550)])
X = data['clips'][kicking_train]
#X = data['clips'][data_kicking_cmu]
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

#feet = np.array([9,10,11,12,13,14,21,22,23,24,25,26])

feet = np.array([24,25,26])


Y = X[:,feet]
#Y shape: 11,3,240
#I = np.arange(len(X))
#rng.shuffle(I)
#X, Y = X[I], Y[I]

batchsize = 1
window = X.shape[2]

network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network_second.load(np.load('network_core.npz'))

network_first = create_regressor(batchsize=batchsize, window=window, input=Y.shape[1])
network = Network(network_first, network_second[1], params=network_first.params)
#network.load(np.load('network_regression_kick.npz'))
E = theano.shared(X, borrow=True)
F = theano.shared(Y, borrow=True)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network, F, E, filename='network_regression_kick.npz')
