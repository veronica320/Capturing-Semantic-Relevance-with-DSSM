##################################################################################################
##                               Providing data for training                                    ##
##                                                                                              ##
##################################################################################################

import re
from keras.utils import np_utils
from collections import Counter
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import operator
import copy
import json

# reading data from data file
def read_data(datafile, train=True):
	lines = list(open(datafile, "r").readlines())
	q_list = []
	lab_list = []
	for line in lines:
		q1, q2, rel = line.strip().split('@@@@@')
		q_list.append((q1, q2))
		lab_list.append(int(rel))
	return q_list, lab_list

# padding text according to maxlen
def w2i(sent, maxlen, w2idic):
	sent = sent.split(' ')
	for i, word in enumerate(sent):
		sent[i] = w2idic[word] if word in w2idic else 0
	return pad_sequences([sent], maxlen=maxlen, padding='post', truncating='post')[0]

# generate pre-trained word vector matrix
def build_w2v(i2v, w2idic, dim):
	vsize = len(w2idic)
	embedding_matrix = np.random.uniform(-0.5, 0.5, (vsize+1, dim))
	for word, i in w2idic.items():
		embedding_vector = i2v.get(word)
		if embedding_vector is not None:
			try:
				embedding_matrix[i] = embedding_vector
			except:
				pass
	return embedding_matrix

# load training and test data
def load_data(datafile, maxlen, nb_class, shuffle, w2v, i2v, w2vdim, w2idic, train=True):
	if train:
		print('Loading training data...')
	else:
		print('Loading test data...')

	q_list, lab_list = read_data(datafile, train)
	nb_class_train = len(set(lab_list))
	nb_train = [(i, Counter(lab_list)[i]) if i in Counter(lab_list) else (i, 0) for i in range(nb_class)]
	nb_train = sorted(nb_train, key=operator.itemgetter(1), reverse=True)

	if shuffle:

		lab_list_shuffle = []
		indices = list(range(len(q_list)))
		np.random.shuffle(indices)
		for i in indices:
			lab_list_shuffle.append(lab_list[i])
		lab_list = lab_list_shuffle

		q_list_shuffle = []
		for i in indices:
			q_list_shuffle.append(q_list[i])
		q_list = q_list_shuffle
		print('Data shuffled!')

	y = np_utils.to_categorical(np.asarray(lab_list), nb_class)
	x1 = []
	x2 = []
	for q_pair in q_list:
		x1.append(w2i(q_pair[0], maxlen, w2idic))
		x2.append(w2i(q_pair[1], maxlen, w2idic))

	print('Data successfully loaded!')
	print('Number of classes in the training data: ' + str(nb_class_train))
	print('Number of samples in each class: ' + str(nb_train))

	if not w2v:
		return [x1, x2], y, 0
	else:
		print('Using pre-trained w2v, calculating weights...')
		word_mat = build_w2v(i2v, w2idic, w2vdim)
		print('w2v weights done!')
		return [x1, x2], y, word_mat