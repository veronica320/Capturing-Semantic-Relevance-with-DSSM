##################################################################################################
##                                        Model structure                                       ##
##                                                                                              ##
##################################################################################################

import numpy as np
from keras import backend
from keras.layers import Embedding, Input, Dropout, Dense, Concatenate, Add, GRU, MaxPooling1D, AveragePooling1D, Flatten
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.core import Activation
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate, dot
from keras.models import Model
from keras import regularizers, optimizers

def create_model(model_name,
                 model_type,
                 nb_class=2,
                 nb_filter=[128, 256, 128],
                 kernel_size=[1, 2, 3],
                 vocab_size=5000,
                 maxlen=20,
                 denseuni=512,
                 dropout=0.2,
                 l2=0.001,
                 lr=0.01,
                 w2v=False,
                 w2vdim=300,
                 w2vmat=None):

	q1_input = Input(shape=(maxlen,), dtype='int32')
	q2_input = Input(shape=(maxlen,), dtype='int32')

	if w2v:
		q1_emb = Embedding(input_dim=vocab_size + 1, output_dim=w2vdim, weights=[w2vmat], input_length=maxlen, trainable=True)(q1_input)
		q2_emb = Embedding(input_dim=vocab_size + 1, output_dim=w2vdim, weights=[w2vmat], input_length=maxlen, trainable=True)(q2_input)
	else:
		q1_emb = Embedding(input_dim=vocab_size + 1, output_dim=w2vdim, input_length=maxlen)(q1_input)
		q2_emb = Embedding(input_dim=vocab_size + 1, output_dim=w2vdim, input_length=maxlen)(q2_input)

	if model_type == 'dssm':

		q1_dense1 = Dense(units=300, activation = "tanh")(q1_emb)
		q1_dense2 = Dense(units=300, activation = "tanh")(q1_dense1)
		q1_pool = Reshape((300,))(MaxPooling1D(pool_size=300, padding='same')(q1_dense2))
		q1_dense3 = Dense(units=128, activation = "tanh")(q1_pool)
		q1_dropout = Dropout(dropout)(q1_dense3)

		q2_dense1 = Dense(units=300, activation = "tanh")(q2_emb)
		q2_dense2 = Dense(units=300, activation = "tanh")(q2_dense1)
		q2_pool = Reshape((300,))(MaxPooling1D(pool_size=300, padding='same')(q2_dense2))
		q2_dense3 = Dense(units=128, activation = "tanh")(q2_pool)
		q2_dropout = Dropout(dropout)(q2_dense3)

		cos_sim = dot([q1_dropout, q2_dropout], axes=1, normalize=True)

		outputs = Dense(units=nb_class, activation='softmax', kernel_regularizer=regularizers.l2(l2))(cos_sim)

		model = Model(inputs=[q1_input, q2_input], outputs=outputs, name=model_name)
		model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=['accuracy'])

		return model

	elif model_type == 'cdssm':
		q1_conv = Conv1D(filters=300, padding='same', kernel_size=1, activation='tanh')(q1_emb)
		q1_pool = Reshape((300,))(MaxPooling1D(pool_size=300, padding='same')(q1_conv))
		q1_dense = Dense(units=128, activation = "tanh")(q1_pool)

		q2_conv = Conv1D(filters=300, padding='same', kernel_size=1, activation='tanh')(q2_emb)
		q2_pool = Reshape((300,))(MaxPooling1D(pool_size=300, padding='same')(q2_conv))
		q2_dense = Dense(units=128, activation = "tanh")(q2_pool)

		cos_sim = dot([q1_dense, q2_dense], axes=1, normalize=True)

		outputs = Dense(units=nb_class, activation='softmax', kernel_regularizer=regularizers.l2(l2))(cos_sim)

		model = Model(inputs=[q1_input, q2_input], outputs=outputs, name=model_name)
		model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=['accuracy'])

		return model
