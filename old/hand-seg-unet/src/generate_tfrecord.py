import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import egohand_utils as ego
import hof_utils as hof
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='name of dataset used', required=False)
args = parser.parse_args()

PATH_TO_SAVE = '/media/aaron/DATA/ubuntu/hand-seg/%s' %args.dataset

if args.dataset == 'egohands':
    def convert(image_paths, out_path):

        # convert data to features
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        print("Converting: " + out_path)

        # open the TFRecords file
        with tf.python_io.TFRecordWriter(out_path) as writer:
            for path in tqdm(image_paths):

                path = path.strip()

                for frame in ego.get_frames(path):

                    # get img and mask
                    img = ego.get_img(path, frame)
                    img = np.reshape(img, (-1, 1))
                    mask = ego.get_mask(path, frame)
                    mask = np.reshape(mask, (-1, 1))

                    # create a feature
                    feature = {'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                               'target': _bytes_feature(tf.compat.as_bytes(mask.tostring()))}

                    # create an example protocol buffer
                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    # serialize to string and write on the file
                    writer.write(example.SerializeToString())

    def create_tfrecord(data_split, out_name):
        data_split = os.path.join(PATH_TO_SAVE, data_split)
        tfrecord_out = os.path.join(PATH_TO_SAVE, '%s.tfrecord' %out_name)
        with open(data_split) as f:
            convert(image_paths=f.readlines(),
                    out_path=tfrecord_out)

    create_tfrecord('data_train.txt', 'train')
    create_tfrecord('data_val.txt', 'val')
    create_tfrecord('data_test.txt', 'test')

if args.dataset == 'hand_over_face':
    def convert(image_list, out_path):

        # convert data to features
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        print("Converting: " + out_path)

        # open the TFRecords file
        with tf.python_io.TFRecordWriter(out_path) as writer:
            for img_id in tqdm(image_list):

                ref_img_id = int(img_id.strip())

                # get img and mask
                img = hof.get_img(ref_img_id)
                img = np.reshape(img, (-1, 1))
                mask = hof.get_mask(ref_img_id)
                mask = np.reshape(mask, (-1, 1))

                # create a feature
                feature = {'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                           'target': _bytes_feature(tf.compat.as_bytes(mask.tostring()))}

                # create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # serialize to string and write on the file
                writer.write(example.SerializeToString())

    def create_tfrecord(data_split, out_name):
        data_split = os.path.join(PATH_TO_SAVE, data_split)
        tfrecord_out = os.path.join(PATH_TO_SAVE, '%s.tfrecord' %out_name)
        with open(data_split) as f:
            convert(image_list=f.readlines(),
                    out_path=tfrecord_out)

    create_tfrecord('data_train.txt', 'train')
    create_tfrecord('data_val.txt', 'val')
    create_tfrecord('data_test.txt', 'test')
