import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import transforms3d.euler as euler
import transforms3d.quaternions as quat
from math import sqrt


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=3, hidden_size=1024, out_frame_size=3):
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size
        
        ##lstm#########################################################
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
        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
     
        out_frame = self.decoder(vec_h2)

        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        print('out_frame:',out_frame.size())
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #in cuda tensor real_seq: b*seq_len*frame_size
    #out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, in_frame):
        in_frame_convert  = torch.autograd.Variable(torch.FloatTensor(in_frame.tolist()).cuda() )
        batch=in_frame_convert.size()[0]

        (vec_h, vec_c) = self.init_hidden(batch)
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())
        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())

        for i in range(26):
            if i==0:
                (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame_convert, vec_h, vec_c)
                out_seq = torch.cat((out_seq, out_frame),1)
            else:
                in_frame=out_frame
                (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
                out_seq = torch.cat((out_seq, out_frame),1)


        #(out_frame, vec_h,vec_c) = self.forward_lstm(in_frame_convert, vec_h, vec_c)
        #out_seq = torch.cat((out_seq, in_frame_convert),1)
        #out_seq = torch.cat((out_seq, out_frame),1)
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        #loss_value=loss.data[0]
        print('loss:',loss.data.cpu().numpy())
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
    #data_kicking_cmu=np.hstack([np.arange(1553, 1555),np.arange(1555, 1558),np.arange(2278, 2281)])
    data_kicking_cmu=np.hstack([np.arange(2278, 2280)])
    #X = data['clips'][kicking_train]
    X = data['clips'][data_kicking_cmu]
    X = np.swapaxes(X, 1, 2).astype(np.float32)
    print('X',X)
    X_1=X.copy()
    #feet = np.array([9,10,11,12,13,14,21,22,23,24,25,26])
    for i in range(X.shape[1]):
        X_1[:,i]=X_1[:,i]+10


    preprocess = np.load('D:/motionsynth_data/synth/preprocess_core.npz')
    print('Xmean:',preprocess['Xmean'])
    print('Xstd:',preprocess['Xstd'])
    X = (X - preprocess['Xmean']) / preprocess['Xstd']
    X_1 = (X_1 - preprocess['Xmean']) / preprocess['Xstd']
    print('X after minus mean:',X)
    feet = np.array([27,28,29])
    gt_joint = np.np.array([27,28,29,24,25,26,21,22,23,18,19,20,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,3,4,5,30,31,32,33,34,35,36,37,38,39,40,41,3,4,5,0,1,2,36,37,38,42,43,44,45,46,47,48,49,50,51,52,53,36,37,38,54,55,56,57,58,59,60,61,62,63,64,65])
    Y = X[:,feet]
    #groundtruth_joints=X[:,0:(total_dof*3)]
    #Y = X[:,0:(total_dof*3)]
    groundtruth_joints=X[:,gt_joint]
    #Y shape: 11,3,240
    print('groundtruth_joints:',groundtruth_joints.shape)
    print('Y shape: ',Y.shape)
    return (Y,groundtruth_joints,X_1)
def motion_chosen(x):
    if x ==0:
        return random.randint(203,212)
    elif x==1:
        return random.randint(116,125)
    else:
        return 1

def test(input_frame, groundtruth_frame, batch,  read_weight_path,  write_bvh_motion_folder):
    torch.cuda.set_device(0)
    
    model = acLSTM()
    
    model.load_state_dict(torch.load(read_weight_path))

    model.cuda()
    total_loss=[]
    total_ijvloss=[]
    

    for iteration in range(100):
        input_batch=[]
        groundtruth_batch=[]
        print('iteration no:',iteration)
        for x in range(batch):
            sample_range=input_frame.shape[0]
            sample_choosing=random.randint(0,sample_range-1)
            sample=input_frame[sample_choosing,:]
            print('sample chosen:',sample.shape)
            #sample=sample.view()
            groundtruth_after_sample=groundtruth_frame[sample_choosing,:]
            #predict_groundtruth= torch.autograd.Variable(torch.FloatTensor(groundtruth_frame[:,1:seq_len+1].tolist())).cuda().view(real_seq_np.shape[0],-1)
            frame_range=input_frame.shape[2]
            #frame_choosing=random.randint(0,frame_range-1)
            frame_choosing=motion_chosen(sample_choosing)
            frame=sample[:,frame_choosing]
            groundtruth_after_frame=groundtruth_after_sample[:,frame_choosing]

            T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
            #xyz?
            R=[0,1,0,(random.random()-0.5)*np.pi*2]
            #print('frame:',type(frame))
            #print('frame shape:',frame.shape[0])
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
        print('predict_joints size:',predict_joints.shape)
        print('predict_joints type:',type(predict_joints))
        loss=model.calculate_loss(predict_joints, predict_groundtruth_seq)
        loss =loss.data.cpu().numpy()
        print('current loss:',loss)
        total_loss.append(loss)
        ijvLoss=ijv_loss(predict_joints,predict_groundtruth_seq)
        print('ijvLoss type:',type(ijvLoss))
        print('ijvLoss:',ijvLoss)
        total_ijvloss.append(ijvLoss)
    total_loss_mean=sum(total_loss)/len(total_loss)
    ijvLossMean=sum(total_ijvloss)/len(total_ijvloss)    
    print('total_loss:',total_loss_mean)
    print('ijv_loss:',ijvLossMean)
    return(predict_joints,predict_groundtruth_seq)

def augment_train_frame_data(train_frame_data, T, axisR) :
    #train_Frame_Data=(102,171)
    print('before train_frame_data:',train_frame_data.shape)

    
    
    mat_r_augment=euler.axangle2mat(axisR[0:3], axisR[3])
    print(train_frame_data.shape[0])
    n=int(train_frame_data[0]/3)
    for i in range(n):
        raw_data=train_frame_data[i*3:i*3+3]
        new_data = np.dot(mat_r_augment, raw_data)+T
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


def plotting(predict_joints,predict_groundtruth_seq,X_1):

    reshape = feet = np.array([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,12,13,14,9,10,11,6,7,8,3,4,5,33,34,35,36,37,38,39,40,41,42,43,44,48,49,50,51,52,53,54,55,56,57,58,59,63,64,65,66,67,68,69,70,71,72,73,74])
    predict_joints=predict_joints[:,reshape]
    predict_groundtruth_seq=predict_groundtruth_seq[:,reshape]
    for x in range(batch):
        #out_seq=np.array(predict_joints[x].data.tolist()).reshape(-1,66)
        out_seq=np.array(predict_joints[x].data.tolist()).reshape(-1,66)
        out_gt_seq=np.array(predict_groundtruth_seq[x].data.tolist()).reshape(-1,66)
        print('out_seq size:', out_seq.shape)
        print('out_seq:', out_seq)

    output_format=np.zeros((1,73))
    output_format[:out_seq.shape[0],:out_seq.shape[1]]=out_seq
    print('output_format:',output_format.shape)
    output=output_format[:,:,np.newaxis]
    print('output:',output.shape)

    output_gt_format=np.zeros((1,73))
    output_gt_format[:out_gt_seq.shape[0],:out_gt_seq.shape[1]]=out_gt_seq
    print('output_format:',output_gt_format.shape)
    output_gt=output_gt_format[:,:,np.newaxis]
    print('output:',output_gt.shape)


    preprocess_data = np.load('D:/motionsynth_data/synth/preprocess_core.npz')
    output = (output * preprocess_data['Xstd']) + preprocess_data['Xmean']
    output_gt = (output_gt * preprocess_data['Xstd']) + preprocess_data['Xmean']
    X_1= (X_1 * preprocess_data['Xstd']) + preprocess_data['Xmean']

    from AnimationPlot import animation_plot

    animation_plot([output,output_gt,X_1], interval=15.15)

def ijv_loss(predict_joints,predict_groundtruth_seq):
    #ijvLoss=torch.dist(predict_joints,predict_groundtruth_seq)
    total_loss=joint_l2_norm(predict_joints,predict_groundtruth_seq)
    return total_loss

    #left leg
def joint_l2_norm(x,y):
    total_loss=[]
    
    for i in np.arange(1,5):
        leftLegLoss=torch.dist(x[:,i*3:i*3+3],x[:,i*3+3:i*3+6])
        leftLegLoss_gt=torch.dist(y[:,i*3:i*3+3],y[:,i*3+3:i*3+6])
        leftLegLoss_final= abs(leftLegLoss- leftLegLoss_gt)
        total_loss.append(leftLegLoss_final)

    for i in np.arange(5,9):
        if i==5:
            rightLegLoss=torch.dist(x[:,3:6],x[:,18:21])
            rightLegLoss_gt=torch.dist(y[:,3:6],y[:,18:21])
            rightLegLoss_final= abs(rightLegLoss- rightLegLoss_gt)
            total_loss.append(rightLegLoss_final)
        else:
            rightLegLoss=torch.dist(x[:,i*3:i*3+3],x[:,i*3+3:i*3+6])
            rightLegLoss_gt=torch.dist(y[:,i*3:i*3+3],y[:,i*3+3:i*3+6])
            rightLegLoss_final= abs(rightLegLoss- rightLegLoss_gt)
            total_loss.append(rightLegLoss_final)

    for i in range(9,13):
        if i==9:
            spineLoss=torch.dist(x[:,3:6],x[:,30:33])
            spineLoss_gt=torch.dist(y[:,3:6],y[:,30:33])
            spineLoss_final= abs(spineLoss- spineLoss_gt)
            total_loss.append(spineLoss_final)
        else:
            spineLoss=torch.dist(x[:,i*3:i*3+3],x[:,i*3+3:i*3+6])
            spineLoss_gt=torch.dist(y[:,i*3:i*3+3],y[:,i*3+3:i*3+6])
            spineLoss_final= abs(spineLoss- spineLoss_gt)
            total_loss.append(spineLoss_final)


    for i in np.arange(13,17):
        if i ==13:
            leftHandLoss=torch.dist(x[:,36:39],x[:,42:45])
            leftHandLoss_gt=torch.dist(y[:,36:39],y[:,42:45])
            leftHandLoss_final=leftHandLoss - leftHandLoss_gt
            total_loss.append(leftHandLoss_final)
        else:
            leftHandLoss=torch.dist(x[:,i*3:i*3+3],x[:,i*3+3:i*3+6])
            leftHandLoss_gt=torch.dist(y[:,i*3:i*3+3],y[:,i*3+3:i*3+6])
            leftHandLoss_final=leftHandLoss - leftHandLoss_gt
            total_loss.append(leftHandLoss_final)

    for i in np.arange(17,21):
        if i ==17:
            rightHandLoss=torch.dist(x[:,36:39],x[:,54:57])
            rightHandLoss_gt=torch.dist(y[:,36:39],y[:,54:57])
            rightHandLoss_final=rightHandLoss- rightHandLoss_gt
            total_loss.append(rightHandLoss_final)
        else:
            rightHandLoss=torch.dist(x[:,i*3:i*3+3],x[:,i*3+3:i*3+6])
            rightHandLoss_gt=torch.dist(y[:,i*3:i*3+3],y[:,i*3+3:i*3+6])
            rightHandLoss_final=rightHandLoss- rightHandLoss_gt
            total_loss.append(rightHandLoss_final)

    total_loss=(sum(total_loss))/21

    return total_loss


    #ijvLoss=list(map(lambda a,b:np.linalg.norm(a-b,'fro'),predict_joints,predict_groundtruth_seq)
    





Y,groundtruth_joints,X_1=preprocess()
batch=1
read_weight_path="../train_coordinate_to_frame_weight/v2/0092000.weight"
write_bvh_motion_folder="../test_bvh_aclstm_indian/"

predict_joints,predict_groundtruth_seq=test(Y, groundtruth_joints, batch, read_weight_path,  write_bvh_motion_folder)
print('predict_joints:',predict_joints.shape)
print('predict_groundtruth_seq',predict_groundtruth_seq.shape)
plotting(predict_joints,predict_groundtruth_seq)