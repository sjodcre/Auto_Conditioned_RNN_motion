import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from Network import Network
from AdamTrainer import AdamTrainer
from network import create_core, create_regressor

def motion_chosen(x):
    if x ==0:
        return np.hstack(np.arange(177,186))
    elif x==1:
        return np.hstack(np.arange(96,105))
    elif x==2:
        return np.hstack(np.arange(176,185))
    elif x==3:
        return np.hstack(np.arange(86,95))
    elif x==4:
        return np.hstack(np.arange(97,106))
    elif x==5:
        return np.hstack(np.arange(166,175))
    elif x==6:
        return np.hstack(np.arange(99,108))
    elif x==7:
        return np.hstack(np.arange(207,216))
    elif x==8:
        return np.hstack(np.arange(121,130))
    elif x==9:
    	return np.hstack(np.arange(203,212))
    else:
        return 0

rng = np.random.RandomState(23455)
#hmd05 is original
#data = np.load('../data/processed/data_hdm05.npz')
data = np.load('../data/processed/data_cmu.npz')
data_kicking = np.hstack([np.arange(199, 246), np.arange(862, 906), np.arange(1582,1640), np.arange(2188,2233), np.arange(2796,2844)])
rng.shuffle(data_kicking)

kicking_train = data_kicking[:len(data_kicking)//2]
kicking_valid = data_kicking[len(data_kicking)//2:]
data_kicking_cmu=np.hstack([1542,1543,1545,1546,1548,1553,1554,1555,1556,2278])
#data_kicking_cmu=np.hstack([np.arange(1541, 1545),np.arange(1545,1548),np.arange(1548, 1550)])
#X = data['clips'][kicking_train]
X = data['clips'][data_kicking_cmu]
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

#feet = np.array([9,10,11,12,13,14,21,22,23,24,25,26])

feet = np.array([27,28,29])


Y = X[:,feet]
#Y shape: 10,3,240
I = np.arange(len(X))
rng.shuffle(I)
X, Y = X[I], Y[I]
test_fn= np.zeros((1,73,10),dtype='float32')
train_fn=np.zeros((1,3,10),dtype='float32')
for a in range(Y.shape[0]):
	example_ind=motion_chosen(a)
	train=Y[a,:]
	test=X[a,:]
	print('train:',train.shape)
	train_post=train[:,example_ind]
	test_post=test[:,example_ind]
	print('train_fn', train_fn.shape)
	print('train post in loop:',train_post.shape)
	#train_fn= np.concatenate((train_fn,train_post),axis=0)
	#np.append(train_fn, np.atleast_3d(train_post), axis=1)
	train_fn=np.concatenate((train_fn,train_post.reshape(1,3,10)))
	test_fn=np.concatenate((test_fn,test_post.reshape(1,73,10)))
	print('testing', train_post)
	print('testing reshape',train_post.reshape(1,3,10))

train_fn=train_fn[1:,:]
test_fn=test_fn[1:,:]
print('train final:',train_fn.shape )
#train_fn shape = 10 (sample),3(xyz),10(frames)
batchsize = 1
window = test_fn.shape[2]
#window = 1
network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network_second.load(np.load('network_core.npz'))

network_first = create_regressor(batchsize=batchsize, window=window, input=train_fn.shape[1])
network = Network(network_first, network_second[1], params=network_first.params)
#network.load(np.load('network_regression_kick.npz'))
E = theano.shared(test_fn, borrow=True)
F = theano.shared(train_fn, borrow=True)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=10000, alpha=0.00001)
trainer.train(network, F, E, filename='network_regression_kick_sam.npz')





def augment_train_frame_data(train_frame_data, T, axisR) :
    #train_Frame_Data=(102,171)
    print('before train_frame_data:',train_frame_data.shape)

    
    
    mat_r_augment=euler.axangle2mat(axisR[0:3], axisR[3])
    print(train_frame_data.shape[0])
    n=int(train_frame_data[0]/3)
    for i in range(n):
        raw_data=train_frame_data[i*3:i*3+3]
        #new_data = np.dot(mat_r_augment, raw_data)+T
        new_data = raw_data+T#ignore rotation
        train_frame_data[i*3:i*3+3]=new_data
    
    return train_frame_data
    
def augment_train_data(train_data, T, axisR):
    print('train_data:',train_data.shape[0])
    result=augment_train_frame_data(train_data,T,axisR)
    result=result.tolist()
    #result=list(map(lambda frame: augment_train_frame_data(frame, T, axisR), train_data))
    #equivalent to casting the function augment_train_frame_data(train_data,T,axisR) and then turn it into a list
    print('result:',np.array(result).shape)
    return np.array(result)