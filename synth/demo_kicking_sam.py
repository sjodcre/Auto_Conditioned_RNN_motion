import sys
import time
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T
import msvcrt as m
#pdb.set_trace()
sys.path.append('../nn')

from Network import Network
from network import create_core, create_regressor
from constraints import constrain, foot_sliding, joint_lengths, multiconstraint

rng = np.random.RandomState(23455)

#data = np.load('../data/processed/data_edin_punching.npz')
data = np.load('../data/processed/data_cmu.npz')
#data = np.load('../data/processed/data_hdm05.npz')
#print('data: ', data.files)
#data:['classes','clips']
#print('data classes shape:',data['classes'].shape)
#data classes shape: 3190,
#print('data clips shape:',data['clips'].shape)
#data clips shape: 3190,240,73
data_kicking = np.hstack([np.arange(199, 246), np.arange(862, 906), np.arange(1582,1640), np.arange(2188,2233), np.arange(2796,2844)])
rng.shuffle(data_kicking)
#data_kicking_cmu=np.hstack([np.arange(1541, 1545),np.arange(1545,1548),np.arange(1548, 1550),np.arange(1553, 1555),np.arange(1555, 1558),np.arange(2278, 2281)])
data_kicking_cmu=np.hstack([2279,1542,1543,1545,1546,1548,1553,1554,1555,1556,2278])
#data_kicking_cmu=np.hstack([np.arange(1553, 1555),np.arange(1555, 1558),np.arange(2278, 2281)])
kicking_train = data_kicking[:len(data_kicking)//2]
kicking_valid = data_kicking[len(data_kicking)//2:]
#print('data_kicking: ',data_kicking.shape)
#data_kicking:242,
#print('kicking_train: ',kicking_train.shape)
#kicking_train:121,
#X = data['clips'][kicking_train]
X = data['clips'][data_kicking_cmu]
print('X shape before swap:', X.shape)
#X shape before swap: 121,240,73
#m.getch()
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
print('Xshape after swap: ',X.shape)
#X shape after swap: 121,73,240
#m.getch()
preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']
#feet= xyz position of feet's heel and toe
#feet = np.array([9,10,11,12,13,14,21,22,23,24,25,26])

feet = np.array([24,25,26])
Y = X[:,feet]
#X[:,feet]=0
Y_pad=np.copy(X)
Y_pad[:,feet]=100
print('X:',X)
print('Y: ',Y)
#Y shape: 121,12,240
#m.getch()
batchsize = 1
window = X.shape[2]

network_first = create_regressor(batchsize=batchsize, window=window, input=Y.shape[1], dropout=0.0)
network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw:x/2)

print('network_second[1]: ',network_second[0])
print('network_second params: ',network_second.params)
#network_second.params=[W,b,W,b]
print('after network_second')
network_second.load(np.load('network_core.npz'))
print('after network_second load')
print('network_second params: ',network_second.params)
#network_second.params=[W,b,W,b]
network = Network(network_first, network_second[1], params=network_first.params)
print('after creating a network made of previous networks')
print('network params: ',network.params)
network.load(np.load('network_regression_kick.npz'))
print('after loading network_regression_kick')
print('network type: ',type(network))

test1 = np.load('network_core.npz')
print('network_core.npz: ', test1.files)
test = np.load('network_regression_kick.npz')
print('network_regression_kick.npz: ', test.files)
#network_core.npz: ', ['L001_L002_W', 'L000_L001_W', 'L001_L003_b', 'L000_L002_b'])
#('network_regression_kick.npz: ', ['L000_L006_b', 'L000_L002_b', 'L000_L005_W', 
#'L000_L009_W', 'L001_L002_W', 'L000_L010_b', 'L000_L001_W', 'L001_L003_b']
from AnimationPlot import animation_plot
print('len x: ',len(X))
#len x:121
for i in range(len(X)):
    print('i: ',i)
    #i:0
    print('Y shape: ',Y[i:i+1].shape)
    #Y shape:(1,12,240)
    network_func = theano.function([], network(Y[i:i+1]))
    Y_pad_ori=np.array(Y_pad[i:i+1])
    Xorig = np.array(X[i:i+1])
    print('X shape: ',X.shape)
    #X shape: 121,73,240
    print('Xorig shape: ',Xorig.shape)
    #Xorig shape : 1,73,240  
    start = time.clock()
    print('before network_func')
    Xrecn = network_func()
    # meaning that Y is inserted into network function then output will be Xrecn
    print('Xrecn shape: ',Xrecn.shape)
    #Xrecn shape: (1,73,240)
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Y_pad_ori=(Y_pad_ori* preprocess['Xstd']) + preprocess['Xmean']
    print
    print('before constrain')
    #here Xrecn already same as Xorig, but next is add constraint
    Xrecn = constrain(Xrecn, network_second[0], network_second[1], preprocess, multiconstraint(
        foot_sliding(Xrecn[:,-4:].copy()),
        joint_lengths()), alpha=0.01, iterations=50)
    #print(data_kicking_cmu[i])
    animation_plot([Xorig], interval=15.15)

