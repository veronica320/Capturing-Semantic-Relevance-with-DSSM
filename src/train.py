##################################################################################################
##                                       Training model                                         ##
##                                                                                              ##
##################################################################################################

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from keras.utils import plot_model, np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import model_structure
from collections import Counter
from conf import *
from data_provider import load_data
import numpy as np
import os

# 获取数据中每个类别的比重
def get_class_weights(y, smooth_factor=0.0):
    if smooth_factor > 9999:
        return 'None'
    else:
        y = [np.argmax(i) for i in y]
        counter = Counter(y)

        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for k in counter.keys():
                counter[k] += p

        majority = max(counter.values())
        weight = {cls: float(majority / count) for cls, count in counter.items()}

        return weight

# 训练模型
def train_model(nb_class, trainfile, valfile, model_path, model_name, epoch):

    x, y, w2vmat = load_data(trainfile, maxlen, nb_class, shuffle, w2v, i2v, w2vdim, w2idic, train=True)

    if valfile:
        x_v, y_v, w2vmat_v = load_data(valfile, maxlen, nb_class, shuffle, w2v, i2v, w2vdim, w2idic, train=True)
        val = [x_v, y_v]
    else:
        val = None

    print('Building base model: {}'.format(model_name))

    model = model_structure.create_model(model_name, model_type, nb_class, nb_filter, kernel_size, vocab_size, maxlen, denseuni, dropout, l2, lr, w2v, w2vdim, w2vmat)
    plot_model(model, show_shapes=True, to_file='model_diagram/'+model_name+'.png')

    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    checkpoint = ModelCheckpoint(model_path + '/' + model_name + '.model', monitor=monitor, verbose=1, save_weights_only=False, save_best_only=True, mode='auto')

    earlystop = EarlyStopping(monitor=monitor, patience=patience, verbose=1, mode='auto')

    print('Begin training model: {}'.format(model_name))


    history = model.fit(x, y,
                        validation_data=val,
                        batch_size=batch_size,
                        epochs=epoch,
                        callbacks=[checkpoint, earlystop],
                        validation_split=split,
                        class_weight=get_class_weights(y, smooth_factor),
                        verbose=2)

    print('Model successfully trained: {}'.format(model_name))

    return model, history

# 绘制训练过程中模型的表现
def plot_training(history, name):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = [i + 1 for i in range(len(acc))]
    with PdfPages(name + '.pdf') as pdf:
        plt.figure()
        plt.subplot(211)
        plt.plot(epochs, loss, 'r-', label='train_loss')
        plt.plot(epochs, val_loss, 'b-', label='val_loss')
        plt.ylabel('loss')
        plt.legend(loc='upper right')

        plt.subplot(212)
        plt.plot(epochs, acc, 'r:', label='train_acc')
        plt.plot(epochs, val_acc, 'b:', label='val_acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc='upper left')

        pdf.savefig()
        plt.close()


#######################################################################################################################

# 训练模型并绘制表现
model, history = train_model(nb_class, trainfile, valfile, model_path, model_name, epoch)
plot_training(history, model_path + '/' + model_name)
