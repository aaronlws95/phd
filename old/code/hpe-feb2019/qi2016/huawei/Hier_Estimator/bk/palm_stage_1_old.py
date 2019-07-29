__author__ = 'QiYE'

from keras.models import Model,model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import h5py
import keras

import numpy
import os
import tensorflow as tf
from ..Model import multi_resolution
from . import trainer
from ..utils import get_err
os.environ["CUDA_VISIBLE_DEVICES"]="3"
setname='mega'
#
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.45):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())


_EPSILON = 10e-8

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
num_kern=[32,64]

lr=0.00003
palm_jnt=3
version = 'palmjnt%d_s0_jiter_ker%d_lr%f'%(palm_jnt,num_kern[0],lr)
# source_dir = 'F:/HuaweiProj/data/mega'
# save_dir = 'F:/HuaweiProj/data/mega'
source_dir='/media/Data/Qi/data'
save_dir='/media/Data/Qi/data'

palm_idx =[0,1,5,9,13,17]
batch_size=128


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses=[]

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        numpy.save('%s/detector/loss_history_%s'%(save_dir,version),[self.losses,self.val_losses])


def load_train_data(save_dir):

    f = h5py.File('%s/source/train_crop_norm_v1.h5'%save_dir, 'r')
    valid_idx =f['valid_idx'][...]
    train_x0 = f['img0'][...][valid_idx]
    train_y= f['uvd_norm_gt'][...][:,palm_idx,:][valid_idx]
    f.close()
    print('trainset loaded',train_x0.shape,train_y.shape)

    normuvd = numpy.load("%s/hier/result/train_normuvd_palm_s0_rot_scaleker32_lr0.000030.npy"%(save_dir))[valid_idx]
    numimg=valid_idx.shape[0]-128
    return numpy.expand_dims(train_x0[:numimg],axis=-1), train_y[:numimg],normuvd[:numimg]

def load_test_data(save_dir):

    f = h5py.File('%s/hier/result/test_crop_for_palm.h5'%save_dir, 'r')
    test_x0 = f['img0'][...][:,palm_jnt,:]
    test_x1 = f['img1'][...][:,palm_jnt,:]
    test_y= f['offset'][...][:,palm_jnt,:]
    f.close()
    print('testset loaded',test_x0.shape,test_y.shape)

    return test_x0,test_x1,test_y

def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # with tf.device('/gpu:1'):


    model = multi_resolution.get_multi_reso_for_finger(img_rows=48,img_cols=48,
                                             num_kern=num_kern,num_f1=1024,num_f2=1024,cnn_out_dim=3)
    model.compile(optimizer=Adam(lr=lr),loss='mean_squared_error')
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s/hier/%s.json"%(save_dir,version), "w") as json_file:
        json_file.write(model_json)
    train_x0, train_y,train_pred_palm= load_train_data(save_dir)
    test_x0,test_x1,test_target = load_test_data(save_dir)
    trainer.fit_palm_stage_1(batch_size=batch_size,n_epochs=1000,model=model,
          train_img0=train_x0,train_target=train_y,train_pred_palm=train_pred_palm,
          test_img0=test_x0,test_img1=test_x1,test_target=test_target,
          model_filepath='%s/hier/weight_%s'%(save_dir,version),
          history_filepath='%s/hier/history_%s'%(save_dir,version),palmjnt=palm_jnt)


def predict():
    # source_dir = 'F:/HuaweiProj/data/mega'
    # save_dir = 'F:/HuaweiProj/data/mega'
    # load json and create model
    version = 'palm_s0_rot_scaleker32_lr0.000030'
    test_loss = numpy.load("%s/hier/model/history_%s.npy"%(save_dir,version))[-1]
    print(numpy.min(test_loss))
    json_file = open("%s/hier/model/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/hier/model/weight_%s"%(save_dir,version))

    x0,x1,x2,y = load_test_data(save_dir)

    loaded_model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')
    normuvd = loaded_model.predict(x={'input0':x0,'input1':x1,'input2':x2},batch_size=batch_size)
    print('shape of pred and ori',y.shape,normuvd.shape)
    normuvd.shape=(normuvd.shape[0],len(palm_idx),3)
    numpy.save("%s/hier/result/train_normuvd_%s"%(save_dir,version),normuvd)
    # normuvd = numpy.load("%s/hier/result/normuvd_%s.npy"%(save_dir,version))
    xyz_pred, xyz_gt, xyz_err = get_err.get_err_from_normuvd(base_dir=save_dir,dataset='train',normuvd=normuvd,jnt_idx=palm_idx,setname='mega')



if __name__ == '__main__':
    # train()
    train()
    # train()
