'''
This is the main script implementing the codebook learning solution. It could be used for
both LOS and NLOS situations. The difference is basically in the training and testing
data.
INPUTS: A data preparing function should be added to load and pre-process the training
        and testing data. The processed data should take the following format:
    - train_inp: is a numpy array of input channel vectors for training.
                 Size is (# of training samples X # of antennas).
    - train_out: is a numpy vector with EGC targets. Size is (# of training samples,)
    - val_inp: is a numpy array of input channel vectors for testing.
               Size is (# of testing samples X # of antennas).
    - val_out: is a numpy vector with EGC targets. Size is (# of testing samples,).
OUTPUTS:
    The codebook is saved at the end of the file

Author: Muhammad Alrabeiah
Sept. 2019
'''
import numpy as np
from keras import optimizers
from keras.models import Input, Model
from complex_fc import CompFC
from auxiliary import PowerPooling
import scipy.io as scio
from DataPrep import dataPrep # Example of data preparing function

num_of_beams = [2, 4, 8, 16, 32, 64, 96, 128]

# Training and testing data:
# --------------------------

batch_size = 500
#-------------------------------------------#
# Here should be the data_preparing function
# It is expected to return:
# train_inp, train_out, val_inp, and val_out
#-------------------------------------------#
len_inp = train_inp.shape[1]
for N in num_of_beams:
    print(str(N) + '-beams Codebook')

    # Model:
    # ------
    xBatch = Input(shape=(len_inp,))
    fc1 = CompFC(N, seed=None, scale=np.sqrt(64), activation='linear')(xBatch)
    max_pooling = PowerPooling(2 * N)(fc1)
    model = Model(inputs=xBatch, outputs=max_pooling)

    # Training:
    # ---------
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    model.compile(optimizer=adam, loss='mse')
    model.fit(train_inp, train_out,
              epochs=100,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(val_inp, val_out))


    # Extract learned codebook:
    # -------------------------
    theta = np.array(model.get_weights()[0])
    print(theta.shape)
    name_of_file = 'theta_NLOS' + str(N) + 'vec.mat'
    scio.savemat(name_of_file,
                 {'train_inp': train_inp,
                  'train_out': train_out,
                  'val_inp': val_inp,
                  'val_out': val_out,
                  'codebook': theta})
