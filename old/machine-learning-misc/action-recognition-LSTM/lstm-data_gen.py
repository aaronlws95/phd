import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import pad_sequences
import keras.backend.tensorflow_backend as KTF
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

root_path = '/media/aaron/DATA/ubuntu/fpa-benchmark/'

parser = argparse.ArgumentParser()
parser.add_argument('--run_opt', type=int, default=2, choices=[1, 2, 3], help='1 to save, 2 to load, 3 for error', required=True)
parser.add_argument('--root_path', type=str, default=root_path, help='root path of files')
args = parser.parse_args()

video_path = os.path.join(root_path, 'Video_files')
ckpt_path = os.path.join(root_path, 'checkpoint')

#GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# def get_session(gpu_fraction, show_log=False):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=show_log))
# KTF.set_session(get_session(gpu_fraction = 1))

#PARAMS
batch_size = 2
num_frames = 10
dim = (32, 32, 3)
num_classes = 45
epochs = 50
num_hidden = 100
drop_rate = 0.2

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path):
        self.current_idx = 0
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.dim = dim
        self.num_classes = num_classes
        self.on_epoch_end()
        self.data_list = self.__get_data_list(data_path)

    def __len__(self):
        return len(self.data_list)//self.batch_size

    def __getitem__(self, index):
        x, y = self.__generate()
        return x, y

    def get_target_list(self):

        target_length = (len(self.data_list)//self.batch_size)*self.batch_size
        target_list = np.zeros(target_length)
        for i in range(target_length):
            words = self.data_list[i].split()
            target_list[i] = words[1]

        return target_list

    def on_epoch_end(self):
        self.current_idx = 0

    def __get_data_list(self, data_path):
        with open(data_path) as f:
            return f.readlines()

    def __generate(self):
        x = []
        y = np.zeros((self.batch_size, 1))

        while True:
            for i in range(self.batch_size):
                batch_data = self.data_list[self.current_idx].split()
                y[i] = int(batch_data[1])
                img_path = os.path.join(video_path, batch_data[0])
                img_path = os.path.join(img_path, 'color')
                num_imgs = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
                tmp_x = []
                for j in range(num_imgs):
                    if j < num_frames:
                        img = cv2.imread(os.path.join(img_path, 'color_%04d.jpeg' %j))
                        img = cv2.resize(img, (dim[0], dim[1]))
                        img = np.reshape(img, self.dim[0]*self.dim[1]*self.dim[2])
                        img = img.astype(np.float32)
                        img = (img - np.mean(img))/(np.std(img))
                        tmp_x.append(img)
                x.append(tmp_x)
                self.current_idx += 1
            x = pad_sequences(x, maxlen=self.num_frames)
            return x, to_categorical(y, num_classes=self.num_classes)

class Baseline_LSTM():
    def __init__(self):
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.model = self.build_model()
        self.callback_list = self.build_callbacks()
        self.epochs = epochs

    def build_model(self):
         model = Sequential()
         model.add(LSTM(num_hidden, input_shape=(num_frames, dim[0]*dim[1]*dim[2])))
         model.add(Dropout(drop_rate))
         model.add(Dense(num_classes, activation='softmax'))
         model.add(Activation('softmax'))
         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
         model.summary()
         return model

    def build_callbacks(self):
        checkpoint = ModelCheckpoint(filepath= os.path.join(ckpt_path, 'model-{epoch:02d}.hdf5'), monitor='loss', verbose=1)
        tensorboard = TensorBoard(ckpt_path, batch_size=self.batch_size)
        tensorboard.set_model(self.model)
        callback_list = [checkpoint, tensorboard]
        return callback_list

    def train(self, generator, val_generator):
        train_model = self.model.fit_generator(generator=generator, validation_data=val_generator, epochs=self.epochs, callbacks=self.callback_list)

        loss_history = train_model.history['loss']
        acc_history = train_model.history['categorical_accuracy']
        val_loss_history = train_model.history['val_loss']
        val_acc_history = train_model.history['val_categorical_accuracy']

        np_loss_history = np.array(loss_history)
        np_accuracy = np.array(acc_history)
        np_val_loss_history = np.array(val_loss_history)
        np_val_accuracy = np.array(val_acc_history)

        np.savetxt(os.path.join(ckpt_path, 'loss_history.txt'), np_loss_history, delimiter=",")
        np.savetxt(os.path.join(ckpt_path, 'acc_history.txt'), np_accuracy, delimiter=",")
        np.savetxt(os.path.join(ckpt_path, 'val_loss_history.txt'), np_val_loss_history, delimiter=",")
        np.savetxt(os.path.join(ckpt_path, 'val_acc_history.txt'), np_val_accuracy, delimiter=",")

    def test(self, model_to_load, generator):
        loaded_model = load_model(model_to_load)

        predict_raw = loaded_model.predict_generator(generator, verbose=1)

        np.savetxt(os.path.join(ckpt_path, 'predictions_raw.txt') , predict_raw)

        predict_label = np.argmax(predict_raw, axis=-1)
        target = generator.get_target_list()
        np.savetxt(os.path.join(ckpt_path, 'predictions.txt') , np.column_stack((predict_label, target)), fmt='%d')

def error():
    predictions = np.loadtxt(os.path.join(ckpt_path, 'predictions.txt'), dtype=np.uint8)
    predict_label = predictions[:, 0]
    target = predictions[:, 1]

    predict_raw = np.loadtxt(os.path.join(ckpt_path, 'predictions_raw.txt'), dtype=np.float32)

    num_data = len(target)
    top1_accuracy = (num_data - len(np.nonzero(predict_label - target)[0]))/num_data
    print('top-1 acc:', top1_accuracy)

    predict_sorted = np.argsort(predict_raw, axis=-1)
    predict_sorted = predict_sorted[:, -5:]
    top5_accuracy = 0
    for i, prediction in enumerate(predict_sorted):
        if target[i] in prediction:
            top5_accuracy += 1
    top5_accuracy = top5_accuracy/num_data
    print('top-5 acc:', top5_accuracy)

    cm = confusion_matrix(y_true = target, y_pred = predict_label)
    df_cm = pd.DataFrame(cm, range(45), range(45))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
    plt.show()

if args.run_opt == 1:
    train_path = os.path.join(root_path, 'data_train.txt')
    train_generator = DataGenerator(train_path)

    val_path = os.path.join(root_path, 'data_val.txt')
    val_generator = DataGenerator(val_path)

    lstm = Baseline_LSTM()
    lstm.train(generator=train_generator, val_generator=val_generator)

elif args.run_opt == 2:

    data_split_path = os.path.join(root_path, 'data_split_test_575.txt')
    data_generator = DataGenerator(data_split_path)
    model_to_load = os.path.join(ckpt_path, "model-50.hdf5")

    lstm = Baseline_LSTM()
    lstm.test(model_to_load, data_generator)

elif args.run_opt == 3:
    error()
