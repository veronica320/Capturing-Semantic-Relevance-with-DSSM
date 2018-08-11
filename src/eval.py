##################################################################################################
##                                       Evaluating model                                       ##
##                                                                                              ##
##################################################################################################

from keras.models import load_model
import numpy as np
import copy
import keras.backend as K
from conf import *
from data_provider import load_data
import time
import operator
from collections import Counter, defaultdict
import os
import tensorflow as tf
from google.protobuf import text_format
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 在测试集上评估模型表现
def eva_model(testfile, nb_class, model_path, model_name, result_path, all_res_file):
	x, y_true, w2vmat = load_data(testfile, maxlen, nb_class, False, w2v, i2v, w2vdim, w2idic, train=False)

	if not os.path.isdir(result_path):
		os.mkdir(result_path)

	ouf = open(result_path + '/' + model_name + '_score.txt', 'w')
	model = load_model(model_path + '/' + model_name + '.model')
	print('Model successfully loaded: {}'.format(model_name))

	all_res_file = open(all_res_file, 'a')
	all_res_file.write(model_name+'\n')

	y_true = np.argmax(y_true, axis=1)
	y_pred = np.argmax(model.predict(x), axis=1)

	res = {}
	res['Acc'] = round(accuracy_score(y_true=y_true, y_pred=y_pred) ,3)
	res['Precision'] = round(precision_score(y_true=y_true, y_pred=y_pred), 3)
	res['Recall'] = round(recall_score(y_true=y_true, y_pred=y_pred), 3)
	res['F-score'] = round(f1_score(y_true=y_true, y_pred=y_pred), 3)

	print(res)

	for key in res:
		string = '{}: {}\t'.format(key, res[key])
		ouf.write(string)
		all_res_file.write(string)

	ouf.write('\n')
	all_res_file.write('\n\n')

#######################################################################################################################

eva_model(testfile, nb_class, model_path, model_name, result_path, all_res_file)