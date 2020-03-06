import tensorflow as tf
import numpy as np
import keras.backend.tensorflow_backend as KTF
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import callbacks
import argparse
import os
from sklearn.metrics import confusion_matrix
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
data_path = os.path.join(root_path, 'data')
eval_path = os.path.join(root_path, 'eval')

#GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# def get_session(gpu_fraction, show_log=False):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=show_log))
# KTF.set_session(get_session(gpu_fraction = 0.5))

#PARAMS
batch_size = 2
num_frames = 10
dim = (32, 32, 3)
num_classes = 45
epochs = 50
num_hidden = 100
drop_rate = 0.2

def create_dataset(filepath):
    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                            "label": tf.FixedLenFeature([], tf.int64)}

        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        # Turn your saved image string into an array
        parsed_features['image'] = tf.decode_raw(
            parsed_features['image'], tf.float32)

        return parsed_features['image'], parsed_features["label"]

    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the batchsize
    dataset = dataset.batch(batch_size)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    image = tf.reshape(image, [-1, num_frames, dim[0]*dim[1]*dim[2]])

    # Create a one hot array for your labels
    label = tf.one_hot(label, num_classes)

    return image, label

def get_labels(data_dir_list, batch_size):
    data_list = []
    with open(data_dir_list) as f:
        data_list = f.readlines()
    labels_length = (len(data_list)//batch_size)*batch_size
    label_list = np.zeros(labels_length)
    for i in range(labels_length):
        words = data_list[i].split()
        label_list[i] = words[1]
    return label_list

def get_num_data(data_dir_list):
    with open(data_dir_list) as f:
        data_list = f.readlines()
        return(len(data_list))

class Baseline_LSTM():
    def __init__(self):
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.dim = dim
        self.epochs = epochs

    def build_model(self, input_frames, labels):

        model_input = layers.Input(tensor=input_frames)
        model = models.Sequential()
        model.add(layers.LSTM(self.num_hidden, input_shape=(self.num_frames, self.dim[0]*self.dim[1]*self.dim[2])))
        model.add(layers.Dropout(drop_rate))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.add(layers.Activation('softmax'))
        model.summary()

        model_output = model(model_input)

        model = models.Model(inputs=model_input, outputs=model_output)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['categorical_accuracy'],
                      target_tensors=[labels])
        return model

    def build_callbacks(self, model):
        checkpoint = callbacks.ModelCheckpoint(filepath= os.path.join(ckpt_path, 'model-{epoch:02d}.hdf5'), monitor='loss', verbose=1)
        tensorboard = callbacks.TensorBoard(ckpt_path, batch_size=self.batch_size)
        tensorboard.set_model(model)
        callback_list = [checkpoint, tensorboard]
        return callback_list

    def train(self, frames, labels, num_data, val_data, num_val):
        def save_history(train_model):
            loss_history = train_model.history['loss']
            acc_history = train_model.history['categorical_accuracy']
            val_loss_history = train_model.history['val_loss']
            val_acc_history = train_model.history['val_categorical_accuracy']

            np_loss_history = np.array(loss_history)
            np_accuracy = np.array(acc_history)
            np_val_loss_history = np.array(val_loss_history)
            np_val_accuracy = np.array(val_acc_history)

            np.savetxt(os.path.join(eval_path, 'loss_history.txt'), np_loss_history, delimiter=",")
            np.savetxt(os.path.join(eval_path, 'acc_history.txt'), np_accuracy, delimiter=",")
            np.savetxt(os.path.join(eval_path, 'val_loss_history.txt'), np_val_loss_history, delimiter=",")
            np.savetxt(os.path.join(eval_path, 'val_acc_history.txt'), np_val_accuracy, delimiter=",")

        model = self.build_model(frames, labels)
        callback_list = self.build_callbacks(model)
        train_model = model.fit(epochs=self.epochs,
                  steps_per_epoch=num_data//self.batch_size,
                  callbacks=callback_list,
                  validation_data=val_data,
                  validation_steps=num_val)
        save_history(train_model)

    def test(self, model_to_load, frames, labels, num_data, dataset):
        loaded_model = models.load_model(model_to_load)

        predict_raw = loaded_model.predict(frames,
                                           steps=num_data//self.batch_size,
                                           verbose=1)

        np.savetxt(os.path.join(eval_path, 'predictions_raw_%s.txt' %dataset) , predict_raw)

        predict_label = np.argmax(predict_raw, axis=-1)

        np.savetxt(os.path.join(eval_path, 'predictions_%s.txt' %dataset) , np.column_stack((predict_label, labels)), fmt='%d')

def error(dataset):
    print('Error:', dataset)
    predictions = np.loadtxt(os.path.join(eval_path, 'predictions_%s.txt' %dataset), dtype=np.uint8)
    predict_label = predictions[:, 0]
    target = predictions[:, 1]

    predict_raw = np.loadtxt(os.path.join(eval_path, 'predictions_raw_%s.txt' %dataset), dtype=np.float32)

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

def plot_metrics(metric_files):
    def plot_graph(file):
        fig, ax = plt.subplots()
        y = np.loadtxt(os.path.join(eval_path, '%s.txt' %file))
        ax.plot(range(len(y)), y, label=file)
        ax.set(title=file)
        plt.show()

    for file in metric_files:
        plot_graph(file)

def epoch_acc_history(frames, labels, num_data, dataset, max_epoch):
    top1_accuracy_list = []
    top5_accuracy_list = []
    for epoch in range(1, max_epoch+1):
        model_to_load = os.path.join(ckpt_path, 'model-%02d.hdf5' %epoch)
        loaded_model = models.load_model(model_to_load)

        predict_raw = loaded_model.predict(frames,
                                           steps=num_data//batch_size,
                                           verbose=1)

        np.savetxt(os.path.join(eval_path, 'predictions_raw_%s_epoch%s.txt' %(dataset, epoch)) , predict_raw)

        predict_label = np.argmax(predict_raw, axis=-1)

        np.savetxt(os.path.join(eval_path, 'predictions_%s_epoch%s.txt' %(dataset, epoch)) , np.column_stack((predict_label, labels)), fmt='%d')

        predictions = np.loadtxt(os.path.join(eval_path, 'predictions_%s_epoch%s.txt' %(dataset, epoch)), dtype=np.uint8)
        predict_label = predictions[:, 0]
        target = predictions[:, 1]

        predict_raw = np.loadtxt(os.path.join(eval_path, 'predictions_raw_%s_epoch%s.txt' %(dataset, epoch)), dtype=np.float32)

        top1_accuracy = (num_data - len(np.nonzero(predict_label - target)[0]))/num_data
        top1_accuracy_list.append(top1_accuracy)

        predict_sorted = np.argsort(predict_raw, axis=-1)
        predict_sorted = predict_sorted[:, -5:]
        top5_accuracy = 0
        for i, prediction in enumerate(predict_sorted):
            if target[i] in prediction:
                top5_accuracy += 1
        top5_accuracy = top5_accuracy/num_data
        top5_accuracy_list.append(top5_accuracy)

        print('epoch [', epoch, ']:', 'top-1 acc:', top1_accuracy, 'top-5 acc:', top5_accuracy)
    np.savetxt(os.path.join(eval_path, 'test_top1_acc.txt') , top1_accuracy_list)
    np.savetxt(os.path.join(eval_path, 'test_top5_acc.txt') , top5_accuracy_list)

if args.run_opt == 1:
    train_tf = os.path.join(data_path, 'train.tfrecord')
    train_data = os.path.join(data_path, 'data_train.txt')
    frames, labels = create_dataset(train_tf)

    val_tf = os.path.join(data_path, 'val.tfrecord')
    val_data = os.path.join(data_path, 'data_val.txt')
    val_frames, val_labels = create_dataset(val_tf)

    lstm = Baseline_LSTM()
    lstm.train(frames, labels, get_num_data(train_data), (val_frames, val_labels), get_num_data(val_data))

elif args.run_opt == 2:
    model_to_load = os.path.join(ckpt_path, "model-50.hdf5")

    test_tf = os.path.join(data_path, 'test.tfrecord')
    test_data = os.path.join(data_path, 'data_test.txt')
    frames, _ = create_dataset(test_tf)
    labels = get_labels(test_data, batch_size)
    lstm = Baseline_LSTM()
    lstm.test(model_to_load, frames, labels, get_num_data(test_data), 'test')

elif args.run_opt == 3:
    # error('test')

    # test_tf = os.path.join(data_path, 'test.tfrecord')
    # test_data = os.path.join(data_path, 'data_test.txt')
    # frames, _ = create_dataset(test_tf)
    # labels = get_labels(test_data, batch_size)
    # epoch_acc_history(frames, labels, get_num_data(test_data), 'test', 50)

    plot_metrics(['loss_history',
                 'acc_history',
                 'val_loss_history',
                 'val_acc_history',
                 'test_top1_acc',
                 'test_top5_acc'])

