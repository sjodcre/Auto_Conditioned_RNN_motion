import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
Condition_num=5
Groundtruth_num=5
In_frame_size = Joints_num*3


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171):
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
    
    #in_frame b*In_frame_size
    #vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    #out_frame b*In_frame_size
    #vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):

        
        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
     
        out_frame = self.decoder(vec_h2) #out b*150
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    
    #in cuda tensor real_seq: b*seq_len*frame_size
    #out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        #real_seq dim = 32,100,171
        batch=real_seq.size()[0]
        #batch =32
        seq_len=real_seq.size()[1]
        #seq_len=100
        condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        print('condition_lst: ',len(condition_lst))
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        #vec_h=[(32,171),(32,171),(32,171)]
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())
        print('out_seq: ',out_seq.size())
        #out_deq =32,1
        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())
        print('out_frame: ',out_frame.size())
        #out_frame=32,171
        print('testing: ',real_seq.size()[2])
        for i in range(seq_len):
            #for each frame
            if(condition_lst[i]==1):##input groundtruth frame
                in_frame=real_seq[:,i]
                #size = 32,171, becaus eonly take one row, depending on i, have 100 row due to seq_len
            else:
                in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
            print('out_seq: ',out_seq.size())
            #[32,172]...then [32,343]...then[32,514], starting column is 0 from initialization
            #the first column is not returned
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss


#numpy array real_seq_np: batch*seq_len*frame_size
def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False, save_bvh_motion=True):
    #train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss, save_bvh_motion)
    #for last and 3rd last parameter, i think if last is True, then write bvh to 3rd last folder
    #dance_batch_np dim =(32,102,171)
    print('data input dim: ',real_seq_np.shape)

    #set hip_x and hip_z as the difference from the future frame to current frame
    dif = real_seq_np[:, 1:real_seq_np.shape[1]] - real_seq_np[:, 0: real_seq_np.shape[1]-1]
    #for all of the batch( 0 to 31), take 2nd dim from 1:end minus 0:end-1, as u can see it is using one less in this dim. e.g. 1:end and 0:end-1
    print('dif: ',dif.shape)
    real_seq_dif_hip_x_z_np = real_seq_np[:, 0:real_seq_np.shape[1]-1].copy()
    #copy original values, from 0: end-1
    real_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    real_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    #as the top there, change the hip_x and hip_z as the difference from the future frame to the current frame
    
    
    real_seq  = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np.tolist()).cuda() )
    #convert back to real_seq
    #real_seq dim = 32,101,171
    print('real_seq: ',real_seq.size())
    seq_len=real_seq.size()[1]-1
    
    #seq_len= 101-1=100
    print('seq_len in train one interation: ',seq_len)
    in_real_seq=real_seq[:, 0:seq_len]
    #in_real_seq dim = 32,100,171
    print('in_real_seq: ', in_real_seq.size())
    #means disregard the last frame?
    
    
    predict_groundtruth_seq= torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np[:,1:seq_len+1].tolist())).cuda().view(real_seq_np.shape[0],-1)
    #the groundtruth that is needed to be predicted, which is the next frame, that is why its 1:seq_len+1 ==1:end for real_seq
    #predict_groundtruth_seq dim = 32,17100
    #.cuda sends the variable to cud, .view changes the dimensions
    print('predict_groundtruth_seq: ',predict_groundtruth_seq.shape)
    
    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)
    #condition num = 5, groundtruth= 5
    
    optimizer.zero_grad()
    #Clears the gradients of all optimized
    
    loss=model.calculate_loss(predict_seq, predict_groundtruth_seq)
    
    loss.backward()
    
    optimizer.step()
    
    if(print_loss==True):
        print ("###########"+"iter %07d"%iteration +"######################")
        print(loss.data.tolist())
        print ("loss: "+str(loss.data.tolist()))

    
    if(save_bvh_motion==True):
        ##save the first motion sequence int the batch.
        gt_seq=np.array(predict_groundtruth_seq[0].data.tolist()).reshape(-1,In_frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(gt_seq.shape[0]):
            gt_seq[frame,Hip_index*3]=gt_seq[frame,Hip_index*3]+last_x
            last_x=gt_seq[frame,Hip_index*3]
            
            gt_seq[frame,Hip_index*3+2]=gt_seq[frame,Hip_index*3+2]+last_z
            last_z=gt_seq[frame,Hip_index*3+2]
        
        out_seq=np.array(predict_seq[0].data.tolist()).reshape(-1,In_frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
            last_x=out_seq[frame,Hip_index*3]
            
            out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
            last_z=out_seq[frame,Hip_index*3+2]
            
        
        read_bvh.write_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_gt.bvh", gt_seq)
        read_bvh.write_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_out.bvh", out_seq)



#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        #length=int(len(dance)/100)
        length = 10
        #length of 10 is used
        if(length<1):
            length=1              
        len_lst=len_lst+[length]

    print('len_lst: ',len_lst)
    
    index_lst=[]
    index=0
    for length in len_lst:
        for i in range(length):
            index_lst=index_lst+[index]
        index=index+1

    print('index_lst: ',index_lst)
    #length is 10, so each dance index is repeated 10 times as it is a constant
    return index_lst

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    #list inside the dance_folder which is inside "../train_data_xyz/indian/"
    dances=[]
    for dance_file in dance_files:
        print ("load "+dance_file)
        dance=np.load(dance_folder+dance_file)
        print ("frame number: "+ str(dance.shape[0]))
        #row (shape0[0]) is no. of frames, column is coordinates in each frame
        dances=dances+[dance]
        #accumulate all the dance into dances
    #print('dances',dances)
    #dances at the end is length of 15, each of the element in the list is an array of the coordinates of the particular dance
    return dances
    
# dances: [dance1, dance2, dance3,....]
def train(dances, frame_rate, batch, seq_len, read_weight_path, write_weight_folder, write_bvh_motion_folder, total_iter=500000):
    #seq_len=100
    seq_len=seq_len+2
    #seqlen(_len=1)
    torch.cuda.set_device(0)
    print('dances size:',len(dances))
    model = acLSTM()

    if(read_weight_path!=""):
        model.load_state_dict(torch.load(read_weight_path))
    
    model.cuda()
    #model=torch.nn.DataParallel(model, device_ids=[0,1])

    current_lr=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    #model.parameters are the weight and bias values for the 3 lstm and the 1 linear

    model.train()
    #set it to training mode, though by default it is already True
    #dance_len_lst contains the index of the dance, the occurance number of a dance's index is proportional to the length of the dance
    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)
    print('length of dance_len_lst: ',dance_len_lst)
    
    speed=frame_rate/30 # we train the network with frame rate of 30
    
    for iteration in range(total_iter):   
        #get a batch of dances
        dance_batch=[]
        for b in range(batch):
            #randomly pick up one dance. the longer the dance is the more likely the dance is picked up
            dance_id = dance_len_lst[np.random.randint(0,random_range)]
            #random based on the length of the dance list, which in this default case is all equally likely as it is a constant 10*15
            dance=dances[dance_id].copy()

            dance_len = dance.shape[0]
            #recall shape[0] is the total frames
            
            start_id=random.randint(10, dance_len-seq_len*speed-10)#the first and last several frames are sometimes noisy. 
            #e.g. randint(10,2212-100*2-10), tbe 100*2 is for the next part (for loop)
            sample_seq=[]
            for i in range(seq_len):
            #i in range(102)
                sample_seq=sample_seq+[dance[int(i*speed+start_id)]]
                #sample_seq=sample_seq+[(0,2,4,...)+start_id]
            
            #augment the direction and position of the dance
            T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
            #xyz?
            R=[0,1,0,(random.random()-0.5)*np.pi*2]
            print('T: ',T)
            print('R: ',R)
            print('sample_seq: ',np.array(sample_seq).shape)
            #sample_seq (a list) length is 102, and each of the 102 is an array of the coordinates
            #he did 102 is because of the augment train data function is it??
            #highly likely 
            sample_seq_augmented=read_bvh.augment_train_data(sample_seq, T, R)
            dance_batch=dance_batch+[sample_seq_augmented]
            print('sample_seq_augmented: ',sample_seq_augmented )
            print(type(sample_seq_augmented))
            #sample_seq_augmented dim = (102,171)
            print('dance_batch: ',len(dance_batch))
            #length is 32 since batch is 32    
        dance_batch_np=np.array(dance_batch)
        print('dance_batch_np: ',dance_batch_np.shape)
        #dance_batch_np dim =(32,102,171)
       
        
        print_loss=False
        save_bvh_motion=False
        if(iteration % 1==0):
            print_loss=True
        if(iteration % 1000==0):
            save_bvh_motion=True
            
        train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss, save_bvh_motion)
        #end=time.time()
        #print end-start
        if(iteration%1000 == 0):
            path = write_weight_folder + "%07d"%iteration +".weight"
            torch.save(model.state_dict(), path)
        

read_weight_path=""
write_weight_folder="../train_weight_aclstm_indian/"
write_bvh_motion_folder="../train_tmp_bvh_aclstm_indian/"
dances_folder = "../train_data_xyz/indian/"
dance_frame_rate=60
batch=32

if not os.path.exists(write_weight_folder):
    os.makedirs(write_weight_folder)
if not os.path.exists(write_bvh_motion_folder):
    os.makedirs(write_bvh_motion_folder)
    

dances= load_dances(dances_folder)

train(dances, dance_frame_rate, batch, 100, read_weight_path, write_weight_folder, write_bvh_motion_folder, 200000)



