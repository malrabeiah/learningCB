import numpy as np
import h5py as h5
import scipy.io as sio


def dataPrep(scale=1,
             EGC=False,
             inputName=None,
             targetName=None,
             valPrec = 0.2,
             save_ind=False,
             load_ind=False,
             out_dim=1,
             inp_dim=128):

    num_ant = 64

    # Loading dataset:
    # ----------------

    with h5.File(inputName,'r') as f:
        fields = [k for k in f.keys()]
        print('Extracting dataset from file: ', inputName)
        rawData = np.array( f['Xstacked'] )
        X = rawData[:,0,:inp_dim]

    if not EGC:
        with h5.File(targetName,'r') as f:
            fields = [k for k in f.keys()]
            print('Extracting dataset from file: ', targetName)
            rawData = np.array( f[fields[0]] )
            t = rawData[:out_dim,:]

    # Making target data:
    # -------------------

    if EGC:
        # Equal gain combining codebook:
        print('Targets with EGC method')
        real_part = X[:,0:num_ant]
        imag_part = X[:,num_ant:]
        sq_real = np.power(real_part, 2)
        sq_imag = np.power(imag_part, 2)
        magnitudes = np.sqrt( sq_imag + sq_real )
        tx = np.sum( magnitudes, axis=1 )
        t = (1/64)*np.power( tx, 2 )
        if len(t.shape) < 2:
            t = t[:, np.newaxis]
        # t = np.reshape(tx, (tx.shape[0], 1, 1))
    else:
        print('Targets with DFT method')
        # DFT codebook received power:
    #	t = np.reshape(t, (t.shape[1], 1, 1)) # 1.31*
        t = 1.5*np.transpose(t)
        print('Target size '+str(t.shape))

    # Training and validation:
    # ------------------------
    print('Creating training and validation inputs')
    numOfTrainSamples = np.floor( (1-valPrec)*X.shape[0] ).astype('int')
    numOfValSamples = np.floor( valPrec*X.shape[0] ).astype('int')
    print('tarining data: '+str(numOfTrainSamples)+'validation data: '+str(numOfValSamples))
    if load_ind:
        shuffled_idx = np.load('shuffled_ind.npy')
        print(shuffled_idx)
    else:
        shuffled_idx = np.random.permutation(X.shape[0])
    if save_ind:
        np.save('shuffled_ind',shuffled_idx)
    X_sh = X[shuffled_idx,:]
    t_sh = t[shuffled_idx,:]
    train_inp = X_sh[0:numOfTrainSamples,:]
    train_out = t_sh[0:numOfTrainSamples,:]
    val_inp = X_sh[numOfTrainSamples:,:]
    val_out = t_sh[numOfTrainSamples:,:]
    if len(val_out.shape) < 2:
        train_out = train_out[:,np.newaxis]
        val_out = val_out[:,np.newaxis]

    return (train_inp, train_out, val_inp, val_out)


# RawData =
# RawData = sio.loadmat('RawData_64ULA_64Sub_rows1650_2050.mat')['rawData']# load rawData as a structured nparray
# fieldsName = RawData.dtype# Read field names in structured nparray R
# print(R.shape)
# print(fieldsName)
# print(fieldsName.names[0])
# print( R[fieldsName.names[1]] )
# print( R[ fieldsName.names[1] ][0,0].shape)
# ch = RawData[ fieldsName.names[0] ][0,0]
# print(ch.shape)
