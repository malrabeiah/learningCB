import numpy as np
import scipy.io as scio
from DataPrep import dataPrep, load_data
from Model import Model

num_ant = 64
num_of_beams = [8, 16, 32, 64, 128]
data_file = 'F:\Dropbox (ASU)\Research\Paper_Codebook Learning\Codes\CodebookLearning_Dataset_Generation\CBL_O1_60_BS3_60GHz_1Path_Corrupted_norm.mat'

batch_size = 500
epoch_num = 10

# Data loading and preparation
train_inp, val_inp = dataPrep(inputName=data_file)
num_train_batch = np.floor(train_inp.shape[0]/batch_size).astype('int')
num_val_batch = np.floor(val_inp.shape[0]/batch_size).astype('int')
limit = num_train_batch*batch_size
train_data = train_inp[0:limit,:] # tailored data
limit = num_val_batch*batch_size
val_data = val_inp[0:limit,:] # tailored data

input_size = train_data.shape[1] # 2*#ant
for num_beams in num_of_beams:
    print(str(num_beams) + '-beams Codebook is generating...')

    # Model:
    net = Model(num_beams, num_ant, mode='recon', accum=True)

    # Training:
    for epoch_idx in range(epoch_num):
        for batch_idx in range(num_train_batch):
            print('beam: %d, batch: %d'%(num_beams, batch_idx))
            for ch_idx in range(batch_size):
                channel = train_data[batch_idx * batch_size + ch_idx, :]
                loss = net.forward(channel)
                # print('Loss: %f' % loss)
                net.backward()
            net.update(lr=0.001) # This statement decides whether accum is True or False

    # Output:
    theta = np.transpose(net.codebook) # To MATLAB format: (#ant, #beams)
    print(theta.shape)
    name_of_file = 'theta_LOS' + str(num_beams) + 'beams.mat'
    scio.savemat(name_of_file,
                 {'train_inp': train_data,
                  'val_inp': val_data,
                  'codebook': theta})





# h_r = np.random.randn(64)
# h_i = np.random.randn(64)
# h = np.concatenate((h_r, h_i), axis=0)
# print(h.shape)
# fc = FullyConnected(4,64)
# a = fc.forward(h)
# print('Break!')