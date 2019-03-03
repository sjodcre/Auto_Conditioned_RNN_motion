import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import read_bvh
import transforms3d.euler as euler
import transforms3d.quaternions as quat



class acLSTM(nn.Module):
    def __init__(self, in_frame_size=3, hidden_size=1024, out_frame_size=63):
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size
        
        ##lstm#########################################################
        #self.lstm = nn.LSTMCell(self.in_frame_size, self.hidden_size)#param+ID
        #self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)#param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    
    #output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        #c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        return  ([h0,h1,h2], [c0,c1,c2])
    

    def forward_lstm(self, in_frame, vec_h, vec_c):

        #
        #vec_h0,vec_c0=self.lstm(in_frame, (vec_h,vec_c))
        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
     
        out_frame = self.decoder(vec_h2)
        #out_frame = torch.cat((out_frame,self.decoder(vec_h0)),1)
        #out_frame = self.decoder(vec_h0)
        #vec_h_new=vec_h0
        #vec_c_new=vec_c0
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        print('out_frame:',out_frame.size())
        
        return (out_frame,  vec_h_new, vec_c_new)
    
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]    

    def forward(self, in_frame,condition_num=5, groundtruth_num=5):
        #in_frame 1,66,1
        in_frame_convert  = torch.autograd.Variable(torch.FloatTensor(in_frame.tolist()).cuda() )
        batch=in_frame_convert.size()[0]
        print('in_frame_convert shape:',in_frame_convert.shape)
        seq_len=in_frame_convert.size()[1]
        #condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)

        (vec_h, vec_c) = self.init_hidden(batch)

        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())
        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())

        #sequence of training: right leg to left leg then to upper bod
        (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame_convert, vec_h, vec_c)
    
        out_seq = torch.cat((out_seq, in_frame_convert),1)
        out_seq = torch.cat((out_seq, out_frame),1)
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss

def preprocess():
    rng = np.random.RandomState(23455)
    #hmd05 is original
    #data = np.load('../data/processed/data_hdm05.npz')
    data = np.load('D:/motionsynth_data/data/processed/data_cmu.npz')
    #data_kicking = np.hstack([np.arange(199, 246), np.arange(862, 906), np.arange(1582,1640), np.arange(2188,2233), np.arange(2796,2844)])
    #rng.shuffle(data_kicking)
    total_dof=22
    #kicking_train = data_kicking[:len(data_kicking)//2]
    #kicking_valid = data_kicking[len(data_kicking)//2:]
    #data_kicking_cmu=np.hstack([1542,1543,1545,1546,1548,1553,1554,1555,1556,2278,2279])
    #data_kicking_cmu=np.hstack([np.arange(1541, 1545),np.arange(1545,1548),np.arange(1548, 1550)])
    data_kicking_cmu=np.hstack([np.arange(1542, 1544),np.arange(1545,1547),np.arange(1548, 1549),np.arange(1553, 1557)])
    #X = data['clips'][kicking_train]
    X = data['clips'][data_kicking_cmu]
    X = np.swapaxes(X, 1, 2).astype(np.float32)

    preprocess = np.load('D:/motionsynth_data/synth/preprocess_core.npz')
    X = (X - preprocess['Xmean']) / preprocess['Xstd']

    #feet = np.array([9,10,11,12,13,14,21,22,23,24,25,26])

    #feet = np.array([27,28,29,24,25,26,21,22,23,18,19,20,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,3,4,5,30,31,32,33,34,35,36,37,38,39,40,41,36,37,38,42,43,44,45,46,47,48,49,50,51,52,53,36,37,38,54,55,56,57,58,59,60,61,62,63,64,65])
    feet = np.array([27,28,29])
    Y = X[:,feet]
    groundtruth_joints=X[:,0:(total_dof*3)]
    #Y shape: 9,3,240
    print('groundtruth_joints:',groundtruth_joints.shape)
    print('Y shape: ',Y.shape)
    return (Y,groundtruth_joints)
def motion_chosen(x):
    if x ==0:
        return random.randint(177,186)
    elif x==1:
        return random.randint(96,105)
    elif x==2:
        return random.randint(174,183)
    elif x==3:
        return random.randint(86,95)
    elif x==4:
        return random.randint(97,106)
    elif x==5:
        return random.randint(164,175)
    elif x==6:
        return random.randint(99,108)
    elif x==7:
        return random.randint(207,216)
    elif x==8:
        return random.randint(121,130)
    else:
        return 1



def train(input_frame, groundtruth_frame,batch,read_weight_path, write_weight_folder, write_bvh_motion_folder, total_iter=100000):
    torch.cuda.set_device(0)
    model = acLSTM()

    if(read_weight_path!=""):
        model.load_state_dict(torch.load(read_weight_path))

    model.cuda()
    current_lr=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    model.train()
    #input_frame = 9,66,240
    for iteration in range(total_iter):
        print(type(input_frame))
        input_batch=[]
        groundtruth_batch=[]
        for x in range(batch):
            sample_range=input_frame.shape[0]
            sample_choosing=random.randint(0,sample_range-1)
            sample=input_frame[sample_choosing,:]
            print('sample chosen:',sample.shape)
            groundtruth_after_sample=groundtruth_frame[sample_choosing,:]
            #predict_groundtruth= torch.autograd.Variable(torch.FloatTensor(groundtruth_frame[:,1:seq_len+1].tolist())).cuda().view(real_seq_np.shape[0],-1)
            #frame_range=input_frame.shape[2]
            #frame_choosing=random.randint(0,frame_range-1)
            frame_choosing=motion_chosen(sample_choosing)
            frame=sample[:,frame_choosing]
            groundtruth_after_frame=groundtruth_after_sample[:,frame_choosing]

            T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
            #T=[(random.random()*1.5),0.0, (random.random()*1.5)]
            #xyz?
            R=[0,1,0,(random.random()-0.5)*np.pi*2]
            print('frame:',type(frame))
            print('frame shape:',frame.shape[0])
            frame=augment_train_data(frame, T, R)
            groundtruth_after_frame=augment_train_data(groundtruth_after_frame, T, R)

            print('frame:',frame)
            input_batch=input_batch+[frame]
            groundtruth_batch=groundtruth_batch+[groundtruth_after_frame]
           #print('predict_joints size:',predict_joints.shape)
            #print('groundtruth_after_frame size:',groundtruth_after_frame.shape)
        input_batch_np=np.array(input_batch)
        groundtruth_batch_np=np.array(groundtruth_batch)
        predict_groundtruth_seq= torch.autograd.Variable(torch.FloatTensor(groundtruth_batch_np.tolist()).cuda() )
        print('input_batch_np: ',input_batch_np.shape)
        predict_joints=model.forward(input_batch_np)
        print('predict_joints: ',predict_joints.shape)
        optimizer.zero_grad()
        loss=model.calculate_loss(predict_joints, predict_groundtruth_seq)
        loss.backward()
        optimizer.step()

        print ("###########"+"iter %07d"%iteration +"######################")
        print(loss.data.tolist())
        print ("loss: "+str(loss.data.tolist()))

        if(iteration%1000 == 0):
            path = write_weight_folder + "%07d"%iteration +".weight"
            torch.save(model.state_dict(), path)


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



read_weight_path=""
write_weight_folder="../train_coordinate_to_frame_weight/v1/"
write_bvh_motion_folder="../train_tmp_bvh_aclstm_indian/"
Y,groundtruth_joints=preprocess()
batch=32
#Y=np.zeros((1,2,3))
#groundtruth_joints=np.zeros((1,2,3))
train(Y, groundtruth_joints,batch,read_weight_path, write_weight_folder, write_bvh_motion_folder,100000)

#train(Y, 60, 32, 100, read_weight_path, write_weight_folder, write_bvh_motion_folder, 200000)