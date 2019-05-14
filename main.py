import numpy as np
from DataFeed import DataFeed
from keras import backend as K
from keras import optimizers
import keras.layers as layer
from keras.models import Sequential, Input, Model
from complexnn.dense import ComplexDense
from MyDense import MyDense
from AuxilLayer import AxuilLayer
from DataPrep import dataPrep
from keras.initializers import Constant
import scipy.io as scio

num_of_beams = [3,4,8,16,32,64]
for N in num_of_beams:
	print(str(N)+'-beams Codebook')
	#data_file = '/home/malrabei/PycharmProjects/CodebookDesign/Proc_RawData_64ULA_64Sub_rows2000_2450.mat'
	#target_file = '/home/malrabei/PycharmProjects/CodebookDesign/targets_64ULA_64Sub_rows2000_2450.mat'
	data_file = '/home/malrabei/PycharmProjects/CodebookDesign/Proc_RawData_64ULA_64Sub_rows1450_1700_2000_2450.mat'
	target_file = '/home/malrabei/PycharmProjects/CodebookDesign/targets_64ULA_64Sub_rows1450_1700_2000_2450.mat'
	# data_file = '/home/malrabei/PycharmProjects/CodebookDesign/PerSample_RawData_64ULA_64Sub_rows1450_1700_2000_2450.mat'
	# target_file = '/home/malrabei/PycharmProjects/CodebookDesign/PerSample_targets_64ULA_64Sub_rows1450_1700_2000_2450.mat'
	batch_size = 500
	train_inpX, train_outX, val_inpX, val_outX = dataPrep(scale=np.sqrt(N), inputName=data_file,targetName=target_file, EGC=True)
	# train_inp=train_inpX[0:51200,:]
	# train_out=train_outX[0:51200,:]
	# val_inp= val_inpX[0:13824,:]
	# val_out= val_outX[0:13824,:]
	#train_inp=train_inpX[0:65000,:]
	#train_out=train_outX[0:65000,:]
	#val_inp= val_inpX[0:16000,:]
	#val_out= val_outX[0:16000,:]
	train_inp=train_inpX[0:100000,:]
	train_out=train_outX[0:100000,:]
	val_inp= val_inpX[0:25000,:]
	val_out= val_outX[0:25000,:]
	#train_feed = DataFeed(train_inp, train_out, batch_size=256)
	#val_feed = DataFeed(val_inp, val_out, batch_size=256)
	print(train_inpX.shape)
	print(train_out.shape)
	print(val_inpX.shape)

	len_inp = train_inp.shape[1]
	len_real = len_inp/2
	len_imag = len_inp/2


	# Model:
	# ------
	xBatch = Input(shape=(len_inp,))
	fc1 = MyDense(N, activation='linear', use_bias=False)(xBatch)# bias_initializer=Constant(value=-.1)
	abs_prePool = AuxilLayer(2*N,batch_size)(fc1)
	max_pooling = layer.MaxPool1D(pool_size=int(N),data_format='channels_first')(abs_prePool)
	model = Model(inputs=xBatch, outputs=max_pooling)

	# Training:
	# ---------
	adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
	model.compile(optimizer=adam, loss='mse')
	model.fit(train_inp, train_out,
				epochs=60,
				batch_size=batch_size,
				shuffle=True,
				validation_data=(val_inp, val_out))
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
	model.compile(optimizer=adam, loss='mse')
	model.fit(train_inp, train_out,
				epochs=40,
				batch_size=batch_size,
				shuffle=True,
				validation_data=(val_inp, val_out))

	theta = np.array( model.get_weights()[0] )

	#biases = np.array( model.get_weights()[1] )
	# Magnitudes = np.sqrt(
	#                      np.power(np.cos( theta ),2) +\
	#                      np.power(np.sin( theta ),2)
	#                     )
	# print('The magnitudes of the learned parameters: \n', Magnitudes)
	# np.save('theta_MeanTarget',theta)
	# print(biases)

	# scio.savemat('theta_64N_scaledW_adjTarg.mat',
	#              {'train_inp': train_inp,
	#               'train_out': train_out,
	#               'val_inp': val_inp,
	#               'val_out': val_out,
	#               'codebook': theta,
	#               'biases': biases})

	# name_of_file = 'theta_set2_persample_'+str(N)+'N_scaledW_EGCTarg_noB.mat'
	# scio.savemat(name_of_file,
	#     	    {'train_inp': train_inp,
	#     	     'train_out': train_out,
	#     	     'val_inp': val_inp,
	#              'val_out': val_out,
	#              'codebook': theta})

	name_of_file = 'theta_set2_perset_'+str(N)+'N_scaledW_EGCTarg_noB.mat'
	scio.savemat(name_of_file,
				{'train_inp': train_inp,
				 'train_out': train_out,
				 'val_inp': val_inp,
				 'val_out': val_out,
				 'codebook': theta})

# Train model:
# # ------------
# print('Training model')
# sgd = optimizers.SGD(lr=0.1, decay=0.0001, momentum=0.9, nesterov=True)
# model.compile(sgd, loss='mean_squared_error')
#
#
# out = model.fit_generator(train_feed, epochs=10, verbose=1, validation_data=val_feed)

# max_pooling = layer.Reshape((1,))(max_pooling)
# real_part = layer.Lambda( (lambda x: x[:,0:int(len_real)]), output_shape=(batch_size,len_real) )(fc1)# K.slice(fc1, 0, len_real)
# imag_part = layer.Lambda( (lambda x: x[:,int(len_real):]), output_shape=(batch_size,len_imag) )(fc1)# K.slice(fc1, len_real, len_imag)
# sq_real = layer.Lambda( (lambda x: K.pow(x,2)), output_shape=(batch_size,len_real) )(real_part)
# sq_imag = layer.Lambda( (lambda x: K.pow(x,2)), output_shape=(batch_size,len_imag) )(imag_part)
# abs_values = layer.Lambda( (lambda x: x[0] + x[1]), output_shape=(batch_size,len_real) )([sq_real, sq_imag])
# max_pooling = layer.MaxPool1D(pool_size=len_real)(abs_values)# Requires a 3D input tensor
# max_pooling = layer.Lambda( (lambda x: K.max(x,axis=1)), output_shape=(batch_size,) )(abs_values)
# # max_value = layer.Lambda( auxLayer, output_shape=(1,) )(fc1)

