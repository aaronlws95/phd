__author__ = 'QiYE'
import matplotlib
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.optimizers import Adam
from keras import backend as K
import h5py
import keras

import numpy
import os
import tensorflow as tf

import copy
from ..Model import multi_resolution
from . import trainer
from ..utils import get_err,hand_utils
from multiprocessing import Process,Manager
import time
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"]="0"
setname='mega'
#
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.4):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))




_EPSILON = 10e-8

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


source_dir = 'F:/HuaweiProj/data/mega'
save_dir = 'F:/HuaweiProj/data/mega'
palm_idx =[0,1,5,9,13,17]
batch_size=128


def load_data(save_dir,dataset):
    f = h5py.File('%s/source/%s_crop_norm_vassi.h5'%(save_dir,dataset), 'r')
    test_x0 = f['img0'][...]
    test_x1 = f['img1'][...]
    test_x2 = f['img2'][...]
    test_y= f['uvd_norm_gt'][...][:,palm_idx,:].reshape(-1,len(palm_idx)*3)
    f.close()
    print(dataset,' loaded',test_x0.shape,test_y.shape)
    return numpy.expand_dims(test_x0,axis=-1),numpy.expand_dims(test_x1,axis=-1),numpy.expand_dims(test_x2,axis=-1),test_y
class RealtimeHandposePipeline(object):

    def __init__(self,save_dir,version):
        self.version = version
        # print(self.model.summary())
        self.sync = Manager().dict(count=0)
    def read_and_detect(self):
        KTF.set_session(get_session())

        print(self.version)
        version=self.version
        print('load model',version)
        # load json and create model
        json_file = open("%s/hier/model/%s.json"%(save_dir,version), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("%s/hier/model/weight_%s"%(save_dir,version))
        model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')
        x=load_data(save_dir,dataset='test')
        while True:
            normuvd = model.predict(x={'input0':x[0],'input1':x[1],'input2':x[2]},batch_size=128)
            # print()
            frm = copy.deepcopy(self.sync)
            count=frm['count']-1
            print('read_and_detect',normuvd.shape,count)
            self.sync.update(count=count)
        print("Exiting read_and_detect...")

    def estimate_and_show(self):
        KTF.set_session(get_session())
        version=self.version
        print(self.version)
        print('load model',version)
        # load json and create model
        json_file = open("%s/hier/model/%s.json"%(save_dir,version), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("%s/hier/model/weight_%s"%(save_dir,version))
        model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')
        x=load_data(save_dir,dataset='test')
        while True:
            normuvd = model.predict(x={'input0':x[0],'input1':x[1],'input2':x[2]},batch_size=128)

            frm = copy.deepcopy(self.sync)
            count=frm['count']
            print('estimate_and_show',normuvd.shape,count)
            # self.sync.update(count=count)
        print("Exiting estimate_and_show...")
    def processVideoThreaded(self):
        """
        Use video as input
        :param device: device id
        :return: None
        """

        print("Create image capture and hand detection process...")
        p = Process(target=self.read_and_detect, args=[])
        p.daemon = True
        print("Create hand poes estimation and show process...")
        c = Process(target=self.estimate_and_show, args=[])
        c.daemon = True
        p.start()
        c.start()

        c.join()
        p.join()
    # def predict_hand(self,x0,x1,x2):
    #     normuvd = self.model.predict(x={'input0':x0,'input1':x1,'input2':x2},batch_size=128)
if __name__ == '__main__':


    # load json and create model
    version = 'vass_palm_s0_rot_scale_ker32_lr0.000100'
    estimator=RealtimeHandposePipeline(save_dir,version)
    estimator.processVideoThreaded()

    # s=time.clock()
    # estimator.predict()
    # estimator.predict()
    # estimator.predict()
    # estimator.predict()
    # estimator.predict()
    # print('sequence time',time.clock()-s)
    # # debug()

