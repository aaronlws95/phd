import tensorflow as tf
import numpy as np
import argparse
import os
import unet
import read_meta as rm
from keras.callbacks import ModelCheckpoint, TensorBoard
import cv2
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.preprocessing import image
import image_utils as imgut

root_path = '/media/aaron/DATA/ubuntu/egohands_data'
video_path = os.path.join(root_path, '_LABELLED_SAMPLES')
ckpt_path = os.path.join(root_path, 'checkpoint')
data_path = os.path.join(root_path, 'data')
eval_path = os.path.join(root_path, 'eval')

parser = argparse.ArgumentParser()
parser.add_argument('--run_opt', type=int, default=2, choices=[1, 2, 3], help='1 to save, 2 to load, 3 for error', required=True)
parser.add_argument('--root_path', type=str, default=root_path, help='root path of files')
args = parser.parse_args()

dim = (256, 256)
epochs = 100
batch_size = 1
augment = False
augment_dim = (224, 224)
exp_no = '3'

#train param
load_epoch = 0 # 0 = no load

#test param
load_model_no = 100

#GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# def get_session(gpu_fraction, show_log=False):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=show_log))
# KTF.set_session(get_session(gpu_fraction = 0.5))

def create_dataset(filepath, to_augment):
    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                            "target": tf.FixedLenFeature([], tf.string)}

        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        # Turn your saved image string into an array
        parsed_features['image'] = tf.decode_raw(
            parsed_features['image'], tf.float32)

        # Turn your saved image string into an array
        parsed_features['target'] = tf.decode_raw(
            parsed_features['target'], tf.uint8)

        return parsed_features['image'], parsed_features["target"]

    def augment_data(image, target, resize_dim):
        seed = np.random.uniform(0, 100)
        image = tf.image.random_flip_left_right(image, seed=seed)
        target = tf.image.random_flip_left_right(target, seed=seed)
        image = tf.image.random_crop(image, [1, resize_dim[0], resize_dim[1], 3], seed=seed)
        target = tf.image.random_crop(target, [1, resize_dim[0], resize_dim[1], 1], seed=seed)
        return image, target

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
    image, target = iterator.get_next()

    # Bring your picture back in shape
    image = tf.reshape(image, [batch_size, dim[0], dim[1], 3])

    # Bring your picture back in shape
    target = tf.reshape(target, [batch_size, dim[0], dim[1], 1])
    target = tf.cast(target, tf.float32)

    if to_augment:
        image, target = augment_data(image, target, augment_dim)

    return image, target

def get_num_data(data_file):
    with open(os.path.join(data_path, data_file)) as f:
        video_list = f.readlines()
        num_data = 0
        for video in video_list:
            num_data += len(rm.get_frames(video.strip()))
        return num_data

def save_imgs(predict_model, data_txt, exp_no):
    save_path = os.path.join(eval_path, exp_no)
    save_path = os.path.join(save_path, dataset)
    with open(data_txt) as f:
        image_path_list = f.readlines()
    predict_num = 0
    for path in image_path_list:
        path = path.strip()
        print(path)
        frames = rm.get_frames(path)
        img_path = os.path.join(video_path, path)
        video_save_folder = os.path.join(save_path, path)
        if not os.path.exists(video_save_folder):
            os.makedirs(video_save_folder)
        for frame in frames:

            target = rm.create_target(path, frame, dim)
            target = target.astype(np.float32)
            input_img = cv2.imread(os.path.join(img_path, 'frame_%04d.jpg' %frame))

            input_img = cv2.resize(input_img, (dim[0], dim[1]))
            # input_img = input_img.astype(np.float32)
            # input_img = np.reshape(input_img, dim[0]*dim[1]*3)
            # input_img = (input_img - np.mean(input_img))/np.std(input_img)
            # input_img = np.reshape(input_img, (dim[0], dim[1], 3))

            cv2.imwrite(os.path.join(video_save_folder, 'frame_%04d_target.png' %frame), target)
            cv2.imwrite(os.path.join(video_save_folder, 'frame_%04d_input.png' %frame), input_img)
            cv2.imwrite(os.path.join(video_save_folder, 'frame_%04d_predict.png' %frame), predict_model[predict_num])
            predict_num += 1

def save_imgs_tfrecord(predict_model, data_batch, data_txt, exp_no):

    save_path = os.path.join(eval_path, exp_no)
    save_path = os.path.join(save_path, dataset)

    with tf.Session() as sess:
        with open(data_txt) as f:
            image_path_list = f.readlines()
        predict_num = 0
        for path in image_path_list:
            path = path.strip()
            print(path)
            frames = rm.get_frames(path)
            img_path = os.path.join(video_path, path)
            video_save_folder = os.path.join(save_path, path)
            if not os.path.exists(video_save_folder):
                os.makedirs(video_save_folder)
            for frame in frames:
                tf_batch = sess.run(data_batch)

                #ASSUMING BATCH_SIZE = 1
                target = tf_batch[1][0]
                input_img = tf_batch[0][0]
                input_img = cv2.normalize(input_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                # input_img = input_img.astype(np.float)*255
                input_img = input_img.astype(np.uint8)

                cv2.imwrite(os.path.join(video_save_folder, 'frame_%04d_target.png' %frame), target)
                cv2.imwrite(os.path.join(video_save_folder, 'frame_%04d_input.png' %frame), input_img)
                cv2.imwrite(os.path.join(video_save_folder, 'frame_%04d_predict.png' %frame), predict_model[predict_num])
                predict_num += 1


if args.run_opt == 1:

    train_tf = os.path.join(data_path, 'train_%d.tfrecord' %dim[0])
    train_data = os.path.join(data_path, 'data_train.txt')
    input_img, targets = create_dataset(train_tf, augment)

    val_tf = os.path.join(data_path, 'val_%d.tfrecord' %dim[0])
    val_data = os.path.join(data_path, 'data_val.txt')
    val_input_img, val_targets = create_dataset(val_tf, augment)

    ckpt_path = os.path.join(ckpt_path, exp_no)
    if load_epoch != 0:
        load_model = os.path.join(ckpt_path, 'model-%02d.hdf5' %load_epoch)
    else:
        load_model = None

    model = unet.unet_base(input_img, targets, load_ckpt=load_model)
    # model = unet.unet_pretrain_encoder(input_img, targets, load_ckpt=load_model)

    checkpoint = ModelCheckpoint(filepath= os.path.join(ckpt_path, 'model-{epoch:02d}.hdf5'), monitor='loss', verbose=1)
    tensorboard = TensorBoard(ckpt_path, batch_size=batch_size)
    tensorboard.set_model(model)
    callback_list = [checkpoint, tensorboard]

    train_model = model.fit(epochs=epochs,
                            steps_per_epoch=get_num_data(train_data)//batch_size,
                            callbacks=callback_list,
                            validation_data=(val_input_img, val_targets),
                            validation_steps=get_num_data(val_data),
                            initial_epoch=load_epoch)

elif args.run_opt == 2:

    dataset = 'val'
    data_tf = os.path.join(data_path, '%s_%d.tfrecord' %(dataset, dim[0]))
    data_txt = os.path.join(data_path, 'data_%s.txt' %dataset)
    input_img, targets = create_dataset(data_tf, True)

    ckpt_path = os.path.join(ckpt_path, exp_no)
    load_model = load_model(os.path.join(ckpt_path, 'model-%02d.hdf5' %load_model_no))
    load_model.summary()
    predict_model = load_model.predict(input_img,
                                       steps=get_num_data(data_txt)//batch_size,
                                       verbose=1)

    # # check
    # for model in predict_model:
    #     imgut.info(model, 'model')
    #     imgut.show(np.reshape(model, dim), 1, 'gray')

    save_imgs_tfrecord(predict_model, (input_img, targets), data_txt, exp_no)

elif args.run_opt == 3:

    dataset = 'train'
    data_tf = os.path.join(data_path, '%s_%d.tfrecord' %(dataset, dim[0]))
    data_txt = os.path.join(data_path, 'data_%s.txt' %dataset)
    next_batch = create_dataset(data_tf, augment)

    with tf.Session() as sess:
        for i in range(get_num_data(data_txt)):
            tf_batch = sess.run(next_batch)
            for j in range(batch_size):
                # img = tf_batch[0][j]
                # imgut.info(img, 'input_img')
                # imgut.show(img, 3)
                # img = tf_batch[1][j]
                # imgut.info(img, 'target')
                # if augment:
                #     imgut.show(np.reshape(img, augment_dim), 1, 'gray')
                # else:
                #     imgut.show(np.reshape(img, dim), 1, 'gray')

                input_img = tf_batch[0][j]
                target = tf_batch[1][j]
                alpha = 0.2
                out_img = (alpha * input_img) + ((1-alpha) * target)
                imgut.show(out_img, 3)
