##################################################################################################
##                            Configuration of model parameters                                 ##
##                                                                                              ##
##################################################################################################

import json

# 数据集路径
datadir = 'camp_dataset2/'

# 是否过滤数据集
clean = ['', 'coarse', 'fine'][0]

# 数据集名称
trainfile = datadir+'sim_question_train'+ '_'+ clean +'.txt'
valfile = None
testfile = datadir+'sim_question_test'+ '_' + clean +'.txt'

# 分类任务类别数
nb_class = 2

# 模型结构
model_type = ['dssm', 'cdssm', 'cgru'][0]

# 最大输入长度
maxlen = 10

# CNN卷积核的filter数目
nb_filter = [128, 256, 128]

# CNN卷积核的大小
kernel_size = [1, 2, 3]

# 全连层的维度
denseuni = 512

# drop out ratio
dropout = 0.1

# l2 regularization ratio
l2 = 0.05

# learning rate
lr = 0.001

# 是否用预训练词向量
w2v = [True, False][0]

# 词向量维数
w2vdim = 300

# 词库文件
with open(datadir+'w2idic.json', 'r') as f:
	w2idic = json.load(f)
vocab_size = len(w2idic)

# 词向量文件
i2v = None
if w2v:
	with open('w2v/wiki.json') as f:
		i2v = json.load(f)

# train/val比例
split = 0.1

# 打乱训练数据
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

# 模型名称及相关路径
model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(clean, model_type, 'w2v' if w2v else 'rand', split, maxlen, batch_size, l2, dropout)
model_path = 'models/' + model_name
result_path = 'results/' + model_name
all_res_file = 'results/all_res.txt'
