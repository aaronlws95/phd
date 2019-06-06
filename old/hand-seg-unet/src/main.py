import tensorflow as tf
import numpy as np
import argparse
import os
import unet
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
import cv2
from keras.models import load_model
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
from custom_loss import *
import egohand_utils as ego

parser = argparse.ArgumentParser()
parser.add_argument('--run_opt', type=int, default=2, choices=[1, 2, 3], help='1 to train, 2 to predict, 3 to plot create_dataset', required=True)
parser.add_argument('--experiment', type=str, default=None, help='experiment directory name', required=False)
parser.add_argument('--load_epoch', type=int, default=0, help='0 to not load', required=False)
parser.add_argument('--data_split', type=str, default='train', choices=['train', 'val', 'test'], help='data to use: train, val or test', required=False)
parser.add_argument('--dataset', type=str, default=None, help='name of dataset used', required=False)
parser.add_argument('--model_type', type=str, default=None, help='type of model used', required=False)
args = parser.parse_args()

if args.model_type:
    model_type = args.model_type
else:
    model_type = args.dataset

PATH_TO_WORKDIR = '/media/aaron/DATA/ubuntu/hand-seg/%s' %args.dataset
PATH_TO_MODELDIR = '/media/aaron/DATA/ubuntu/hand-seg/%s/' %model_type

#GPU
# import keras.backend.tensorflow_backend as KTF
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# def get_session(gpu_fraction, show_log=False):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=show_log))
# KTF.set_session(get_session(gpu_fraction = 0.25))

if args.dataset == 'egohand':
    ori_dim = (720, 1280)
    resize_dim = (256, 256)
    jitter_dim = (224, 224)
if args.dataset == 'hand_over_face':
    ori_dim = (216, 384)
    resize_dim = (256, 256)
    jitter_dim = (224, 224)

# get number of frames from data split
def get_num_data(data_file):
    if args.dataset == 'egohand':
        with open(data_file) as f:
            video_list = f.readlines()
            num_data = 0
            for video in video_list:
                num_data += len(ego.get_frames(video.strip()))
            return num_data
    if args.dataset == 'hand_over_face':
        with open(data_file) as f:
            return len(f.readlines())

    return 0

def create_dataset(filepath, batch_size, to_preprocess=True, to_jitter=False, return_ori=False):
    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                            "target": tf.FixedLenFeature([], tf.string)}

        # load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        # turn your saved image string into an array
        parsed_features['image'] = tf.decode_raw(
            parsed_features['image'], tf.float32)

        # turn your saved image string into an array
        parsed_features['target'] = tf.decode_raw(
            parsed_features['target'], tf.float32)

        return parsed_features['image'], parsed_features["target"]

    def preprocess(image, target, resize):
        # BGR --> RGB
        image = image[:, :, :, ::-1]

        # grayscale image at random
        # def gray_3channel():
        #     image_gray = tf.image.rgb_to_grayscale(image)
        #     image_gray = tf.squeeze(image_gray, -1)
        #     image_gray = tf.stack((image_gray,)*3, axis=-1)
        #     return image_gray
        # def false_fn():
        #     return image
        # to_gray = tf.random.uniform([], 0, 1)
        # image = tf.cond(tf.greater(to_gray, tf.constant(0.5)), gray_3channel, false_fn)

        # grayscale
        # image = tf.image.rgb_to_grayscale(image)

        #resize
        if resize_dim is not None:
            image = tf.image.resize_images(image, resize)

        # standardization
        mean, var = tf.nn.moments(image, axes=[0,1,2])
        image = (image - mean)/tf.sqrt(var)

        # normalization
        # image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

        # expand dimension
        target = tf.expand_dims(target, 3)

        if resize_dim is not None:
            target = tf.image.resize_images(target, resize) #resize

        return image, target

    def jitter(image, target, resize):
        seed = np.random.uniform(0, 100)
        image = tf.image.random_flip_left_right(image, seed=seed)
        target = tf.image.random_flip_left_right(target, seed=seed)
        image = tf.image.random_crop(image, [batch_size, resize[0], resize[1], image.shape[-1]], seed=seed)
        target = tf.image.random_crop(target, [batch_size, resize[0], resize[1], target.shape[-1]], seed=seed)
        return image, target

    # this works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # this dataset will go on forever
    dataset = dataset.repeat()

    # set the batchsize
    dataset = dataset.batch(batch_size)

    # create an iterator
    iterator = dataset.make_one_shot_iterator()

    # create your tf representation of the iterator
    image, target = iterator.get_next()

    # reshape
    image = tf.reshape(image, (batch_size, ori_dim[0], ori_dim[1], 3))
    target = tf.reshape(target, (batch_size, ori_dim[0], ori_dim[1]))

    if return_ori:
        image_RGB = image
        image_RGB = image_RGB[:, :, :, ::-1] #BGR-->RGB
        if resize_dim is not None:
            image_RGB = tf.image.resize_images(image_RGB, resize_dim) #resize

    if to_preprocess:
        image, target = preprocess(image, target, resize_dim)

    if to_jitter:
        image, target = jitter(image, target, jitter_dim)

    target = tf.cast(target > 0, dtype=tf.float32)

    if return_ori:
        return image_RGB, image, target
    else:
        return image, target


class TensorBoardImage(Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path

    def summary_image(self, tensor):
        from PIL import Image
        _, height, width, channel = tensor.shape
        img_arr = K.eval(tensor)
        img_arr = cv2.normalize(img_arr, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
        img_arr = np.squeeze(img_arr, axis=(0,-1))
        image = Image.fromarray(img_arr)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height = height,
                                width = width,
                                colorspace = channel,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs=None):
        # Load image
        predict = self.summary_image(self.model.outputs[0])
        summary = tf.Summary(value=[
                             tf.Summary.Value(tag='predict', image=predict)])
        writer = tf.summary.FileWriter(self.log_path)
        writer.add_summary(summary, epoch)
        writer.close()

if args.run_opt == 1:

    to_jitter = True
    load_epoch = args.load_epoch
    final_epoch = 200
    batch_size = 1
    experiment = args.experiment

    experiment_path = os.path.join(PATH_TO_WORKDIR, experiment)
    model_path = os.path.join(experiment_path, 'models')
    log_path = os.path.join(experiment_path, 'logs')
    predict_path = os.path.join(experiment_path, 'predict')
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        os.makedirs(model_path)
        os.makedirs(log_path)
        os.makedirs(predict_path)

    print('TRAINING MODEL')

    # load dataset
    train_tf = os.path.join(PATH_TO_WORKDIR, 'train.tfrecord')
    train_data = os.path.join(PATH_TO_WORKDIR, 'data_train.txt')
    input_img, targets = create_dataset(train_tf, batch_size, to_preprocess=True, to_jitter=to_jitter)

    val_tf = os.path.join(PATH_TO_WORKDIR, 'val.tfrecord')
    val_data = os.path.join(PATH_TO_WORKDIR, 'data_val.txt')
    val_input_img, val_targets = create_dataset(val_tf, batch_size, to_preprocess=True, to_jitter=to_jitter)

    # create model
    if load_epoch != 0:
        load_weights = os.path.join(model_path, 'model-%02d.hdf5' %load_epoch)
    else:
        load_weights = None
    model = unet.unet_pretrain_encoder(input_img, targets, load_weights=load_weights)

    # callbacks

    checkpoint = ModelCheckpoint(filepath= os.path.join(model_path, 'model-{epoch:02d}.hdf5'), monitor='loss', verbose=1)
    tensorboard = TensorBoard(log_path, batch_size=batch_size)
    tensorboard.set_model(model)
    callback_list = [checkpoint, tensorboard]

    # train
    train_model = model.fit(epochs=final_epoch,
                            steps_per_epoch=get_num_data(train_data)//batch_size,
                            callbacks=callback_list,
                            validation_data=(val_input_img, val_targets),
                            validation_steps=get_num_data(val_data)//batch_size,
                            initial_epoch=load_epoch)

elif args.run_opt == 2:

    def save_predict(predict, save_path):
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('predict', data=predict)
        h5f.close()

    data_split = args.data_split
    to_jitter = False
    load_epoch = args.load_epoch
    batch_size = 1
    experiment = args.experiment

    print('PREDICTING DATA: %s' %data_split)

    # create dataset
    data_tf = os.path.join(PATH_TO_WORKDIR, '%s.tfrecord' %data_split)
    data_txt = os.path.join(PATH_TO_WORKDIR, 'data_%s.txt' %data_split)
    img_ori, img, targets = create_dataset(data_tf, batch_size, to_preprocess=True, to_jitter=to_jitter, return_ori=True)

    # load model
    model_path = os.path.join(PATH_TO_MODELDIR, '%s/models' %experiment)
    model = load_model(os.path.join(model_path, 'model-%02d.hdf5' %load_epoch), custom_objects=custom_obj)
    model.summary()

    # predict
    predict_list = model.predict(img,
                                 steps=get_num_data(data_txt)//batch_size,
                                 verbose=1)

    # save predictions
    save_path = os.path.join(PATH_TO_MODELDIR, '%s/predict/predict_%s_%d_%s.h5' %(experiment, data_split, load_epoch, args.dataset))
    save_predict(predict_list, save_path)

elif args.run_opt == 3:

    data_split = args.data_split
    batch_size = 1

    data_tf = os.path.join(PATH_TO_WORKDIR, '%s.tfrecord' %data_split)
    data_txt = os.path.join(PATH_TO_WORKDIR, 'data_%s.txt' %data_split)
    batch_data = create_dataset(data_tf, batch_size, to_preprocess=True ,to_jitter=True, return_ori=True)

    with tf.Session() as sess:
        for i in range(get_num_data(data_txt)):
            next_batch = sess.run(batch_data)
            for j in range(batch_size):
                img_ori = next_batch[0][j]
                img_ori = cv2.normalize(img_ori, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                img = next_batch[1][j]
                mask = next_batch[2][j]

                if img.shape[-1] == 1:
                    img = np.squeeze(img,-1)
                if mask.shape[-1] == 1:
                    mask = np.squeeze(mask, -1)

                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(img_ori)
                ax[0].set_title('img_ori')
                ax[1].imshow(img)
                ax[1].set_title('img')
                ax[2].imshow(mask, cmap='gray')
                ax[2].set_title('mask')
                plt.show()

