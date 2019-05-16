__author__ = 'QiYE'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from ..utils import get_err,hand_utils
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
num_kern=[32,64,96]

lr=0.0001
# best lr = 0.0001

version = 'tmp_vass_palm_s0_rot_scale_ker%d_lr%f'%(num_kern[0],lr)

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

    # def on_batch_end(self, batch, logs={}):
    #     print(' loss ',batch)


def load_data(save_dir,dataset):
    f = h5py.File('%s/source/%s_crop_norm_v1.h5'%(save_dir,dataset), 'r')
    valid_idx =f['valid_idx'][...]
    print(valid_idx.shape)
    test_x0 = f['img0'][...][valid_idx]
    test_x1 = f['img1'][...][valid_idx]
    test_x2 = f['img2'][...][valid_idx]
    test_y= f['uvd_norm_gt'][...][:,palm_idx,:][valid_idx]
    test_y.shape=(test_x0.shape[0],len(palm_idx)*3)
    f.close()
    print(dataset,' loaded',test_x0.shape,test_y.shape)
    return numpy.expand_dims(test_x0,axis=-1),numpy.expand_dims(test_x1,axis=-1),numpy.expand_dims(test_x2,axis=-1),test_y
def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # with tf.device('/gpu:1'):

    model = multi_resolution.get_multi_reso_for_palm_stage_0(img_rows=96,img_cols=96,
                                             num_kern=num_kern,num_f1=2048,num_f2=1024,cnn_out_dim=18)
    model.compile(optimizer=Adam(lr=lr),loss='mean_squared_error')
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s/hier/%s.json"%(save_dir,version), "w") as json_file:
        json_file.write(model_json)
    # model.load_weights("%s/hier/weight_%s_bf"%(save_dir,version))


    train_x0,train_x1,train_x2, train_y = load_data(save_dir,dataset='train')
    numimg=train_x0.shape[0]
    idx1=list(range(0,403579,1))+list(range(448847,numimg,1))
    idx2=range(403579,448847,1)#the frames belonging to  vassilieos
    trainer.fit_palm_stage_0(batch_size=batch_size,n_epochs=1000,model=model,
          train_img0=train_x0[idx1],train_img1=train_x1[idx1],train_img2=train_x2[idx1],train_target=train_y[idx1],
          test_img0=train_x0[idx2],test_img1=train_x1[idx2],test_img2=train_x2[idx2],test_target=train_y[idx2],
          model_filepath='%s/hier/weight_%s'%(save_dir,version),
          history_filepath='%s/hier/history_%s'%(save_dir,version))



def predict():
    # source_dir = 'F:/HuaweiProj/data/mega'
    # save_dir = 'F:/HuaweiProj/data/mega'
    # load json and create model
    version = 'tmp_palm_s0_rot_scale_ker32_lr0.000100'
    test_loss = numpy.load("%s/hier/history_%s.npy"%(save_dir,version))[-1]
    print(numpy.min(test_loss))
    json_file = open("%s/hier/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/hier/weight_%s"%(save_dir,version))
    loaded_model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')

    for dataset in ['test']:
        test_x0,test_x1,test_x2,test_y=load_data(save_dir,dataset=dataset)

        normuvd = loaded_model.predict(x={'input0':test_x0,'input1':test_x1,'input2':test_x2},batch_size=128)
        print('shape of pred and ori',test_y.shape,normuvd.shape)
        normuvd.shape=(normuvd.shape[0],len(palm_idx),3)
        numpy.save("%s/hier/result/%s_normuvd_%s"%(save_dir,dataset,version),normuvd)
        # normuvd = numpy.load("%s/hier/result/normuvd_%s.npy"%(save_dir,version))
        path='%s/source/%s_crop_norm_v1.h5'%(save_dir,dataset)
        xyz_pred, xyz_gt, xyz_err = get_err.get_err_from_normuvd(path=path,dataset=dataset,normuvd=normuvd,jnt_idx=palm_idx,setname='mega')






if __name__ == '__main__':
    # train()
    predict()

    # debug()

