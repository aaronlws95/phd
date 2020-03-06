import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from keras.preprocessing.sequence import pad_sequences

root_path = '/media/aaron/DATA/ubuntu/fpa-benchmark/'
video_path = os.path.join(root_path, 'Video_files')
data_path = os.path.join(root_path, 'data')

num_frames = 10
dim = (32, 32, 3)

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def convert(image_path_list, labels, out_path):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    print("Converting: " + out_path)
    with tf.python_io.TFRecordWriter(out_path) as writer:
        for i, (path, label) in enumerate(zip(image_path_list, labels)):

            print_progress(count=i, total=len(image_path_list)-1)

            img_path = os.path.join(video_path, path)
            img_path = os.path.join(img_path, 'color')
            num_images = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])

            tmp_x = []
            for j in range(num_images):
                if j < num_frames:
                    img = cv2.imread(os.path.join(img_path, 'color_%04d.jpeg' %j))
                    img = cv2.resize(img, (dim[0], dim[1]))
                    img = np.reshape(img, dim[0]*dim[1]*dim[2])
                    img = img.astype(np.float32)
                    img = (img - np.mean(img))/(np.std(img))
                    tmp_x.append(img)

            actual_frames = len(tmp_x)
            tmp_x = np.asarray(tmp_x)
            tmp_x = np.reshape(tmp_x, (1, actual_frames, dim[0]*dim[1]*dim[2]))

            x = pad_sequences(tmp_x, maxlen=num_frames, dtype=np.float32)

            x = np.reshape(x ,(num_frames, dim[0]*dim[1]*dim[2]))

            feature = { 'image': _bytes_feature(tf.compat.as_bytes(x.tostring())),
                    'label': _int64_feature(label) }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def create_tfrecord(data_list_path, out_name):

    tfrecord_path = os.path.join(data_path, data_list_path)
    tfrecord_out = os.path.join(data_path, '%s.tfrecord' %out_name)
    lines = []
    with open(tfrecord_path) as f:
        lines = f.readlines()
        labels = []
        image_path_list = []
        for line in lines:
            split = line.split()
            image_path_list.append(split[0])
            labels.append(int(split[1]))

        convert(image_path_list=image_path_list,
                labels=labels,
                out_path=tfrecord_out)

create_tfrecord('data_train.txt', 'train')
create_tfrecord('data_val.txt', 'val')
create_tfrecord('data_test.txt', 'test')
