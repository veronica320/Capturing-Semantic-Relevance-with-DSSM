##################################################################################################
##                            Configuration of model parameters                                 ##
##                                                                                              ##
##################################################################################################

import json

# data directory
datadir = 'camp_dataset2/'

# filtering scheme
clean = ['', 'coarse', 'fine'][0]

# dataset name
trainfile = datadir+'sim_question_train'+ '_'+ clean +'.txt'
valfile = None
testfile = datadir+'sim_question_test'+ '_' + clean +'.txt'

# number of classes
nb_class = 2

# architecture
model_type = ['dssm', 'cdssm', 'cgru'][0]

# maximum input l
maxlen = 10

# number of cnn filters
nb_filter = [128, 256, 128]

# cnn kernel size
kernel_size = [1, 2, 3]

# dimensionality of dense layer
denseuni = 512

# drop out ratio
dropout = 0.1

# l2 regularization ratio
l2 = 0.05

# learning rate
lr = 0.001

# whether to use pre-trained word vectors
w2v = [True, False][0]

# dimensionality of word vectors
w2vdim = 300

# word dictionary file
with open(datadir+'w2idic.json', 'r') as f:
	w2idic = json.load(f)
vocab_size = len(w2idic)

# word vectors file
i2v = None
if w2v:
	with open('w2v/wiki.json') as f:
		i2v = json.load(f)

# train/val split proportion
split = 0.1

# whether to shuffle training data
shuffle = True

# batch size
batch_size = 200

# epoch
epoch = 30

# patience for early-stopping
patience = 10

# monitor for early-stopping and checkpoint
monitor = 'val_loss'

# smooth factor for category weight
smooth_factor = 0.1

# model name and paths
model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(clean, model_type, 'w2v' if w2v else 'rand', split, maxlen, batch_size, l2, dropout)
model_path = 'models/' + model_name
result_path = 'results/' + model_name
all_res_file = 'results/all_res.txt'
