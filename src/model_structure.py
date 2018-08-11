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

# 封装了三种模型结构
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

	elif model_type == 'cgru':

		q1_conv1 = []
		for i in range(len(nb_filter)):
			conv = Conv1D(nb_filter[i], padding='same', kernel_initializer="normal", kernel_size=kernel_size[i], activation='relu')(q1_emb)
			q1_conv1.append(conv)
		q1_concat1 = Concatenate()(q1_conv1)
		q1_bn1 = BatchNormalization()(q1_concat1)

		q1_conv2 = []
		for i in range(len(nb_filter)):
			conv = Conv1D(nb_filter[i], padding='same', kernel_initializer="normal", kernel_size=kernel_size[i], activation='relu')(q1_bn1)
			q1_conv2.append(conv)
		q1_concat2 = Concatenate()(q1_conv2)
		q1_bn2 = BatchNormalization()(q1_concat2)

		q1_add = Add()([q1_bn1, q1_bn2])
		q1_grul2r = GRU(units=int(denseuni / 2))(q1_add)
		q1_grur2l = GRU(units=int(denseuni / 2), go_backwards=True)(q1_add)
		q1_grubi = concatenate([q1_grul2r, q1_grur2l], axis=1)

		q1_dropout1 = Dropout(dropout)(q1_grubi)
		q1_dense1 = Dense(units=denseuni, activation='relu', kernel_regularizer=regularizers.l2(l2))(q1_dropout1)
		q1_dropout2 = Dropout(dropout)(q1_dense1)

		q2_conv1 = []
		for i in range(len(nb_filter)):
			conv = Conv1D(nb_filter[i], padding='same', kernel_initializer="normal", kernel_size=kernel_size[i], activation='relu')(q2_emb)
			q2_conv1.append(conv)
		q2_concat1 = Concatenate()(q2_conv1)
		q2_bn1 = BatchNormalization()(q2_concat1)

		q2_conv2 = []
		for i in range(len(nb_filter)):
			conv = Conv1D(nb_filter[i], padding='same', kernel_initializer="normal", kernel_size=kernel_size[i], activation='relu')(q2_bn1)
			q2_conv2.append(conv)
		q2_concat2 = Concatenate()(q2_conv2)
		q2_bn2 = BatchNormalization()(q2_concat2)

		q2_add = Add()([q2_bn1, q2_bn2])
		q2_grul2r = GRU(units=int(denseuni / 2))(q2_add)
		q2_grur2l = GRU(units=int(denseuni / 2), go_backwards=True)(q2_add)
		q2_grubi = concatenate([q2_grul2r, q2_grur2l], axis=1)

		q2_dropout1 = Dropout(dropout)(q2_grubi)
		q2_dense1 = Dense(units=denseuni, activation='relu', kernel_regularizer=regularizers.l2(l2))(q2_dropout1)
		q2_dropout2 = Dropout(dropout)(q2_dense1)

		cos_sim = dot([q1_dropout2, q2_dropout2], axes=1, normalize=True)

		outputs = Dense(units=nb_class, activation='softmax', kernel_regularizer=regularizers.l2(l2))(cos_sim)

		model = Model(inputs=[q1_input, q2_input], outputs=outputs, name=model_name)
		model.compile(optimizer=optimizers.Adam(lr=lr), loss="categorical_crossentropy", metrics=['accuracy'])
		return model
