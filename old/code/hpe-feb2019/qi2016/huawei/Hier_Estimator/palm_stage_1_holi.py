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
from ..utils import get_err,data_augmentation

import matplotlib.pyplot as plt
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
num_kern=[32,64,128]
lr=0.0001

version = 'palm_s1_smalljiter_ker%d_lr%f'%(num_kern[0],lr)
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


def load_data(save_dir,dataset):

    f = h5py.File('%s/source/%s_crop_norm_vassi.h5'%(save_dir,dataset), 'r')
    train_x0 = f['img0'][...]
    train_uvd_centre=f['uvd_hand_centre'][...]
    train_y= f['uvd_norm_gt'][...][:,palm_idx,:]
    f.close()

    normuvd = numpy.load("%s/hier/result/%s_normuvd_tmp_vass_palm_s0_rot_scale_ker32_lr0.000100.npy"%(save_dir,dataset))
    print(dataset, ' train_x0.shape,train_y.shape,normuvd.shape',train_x0.shape,train_y.shape,normuvd.shape)

    return numpy.expand_dims(train_x0,axis=-1), train_y,normuvd,train_uvd_centre


def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = multi_resolution.get_multi_reso_for_palm_stage_0(img_rows=96,img_cols=96,
                                             num_kern=num_kern,num_f1=2048,num_f2=1024,cnn_out_dim=18)
    model.compile(optimizer=Adam(lr=lr),loss='mean_squared_error')
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s/hier/%s.json"%(save_dir,version), "w") as json_file:
        json_file.write(model_json)
    train_x0, train_y,train_pred_palm,train_uvd_centre= load_data(save_dir,dataset='train')
    test_x0,test_y,test_pred_palm,test_uvd_centre = load_data(save_dir,dataset='test')
    trainer.fit_palm_stage_1_holi(batch_size=batch_size,n_epochs=1000,model=model,
          train_img0=train_x0,train_target=train_y,train_pred_palm=train_pred_palm,train_uvd_centre=train_uvd_centre,
          test_img0=test_x0,test_target=test_y,test_pred_palm=test_pred_palm,test_uvd_centre=test_uvd_centre,
          model_filepath='%s/hier/weight_%s'%(save_dir,version),
          history_filepath='%s/hier/history_%s'%(save_dir,version))

def predict():
    version='palm_s1_jiter_ker32_lr0.000100'
    test_loss = numpy.load("%s/hier/history_%s.npy"%(save_dir,version))[-1]
    print(numpy.min(test_loss))
    json_file = open("%s/hier/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')
    model.load_weights("%s/hier/weight_%s"%(save_dir,version))


    batch_size=128
    for dataset in ['test']:
        train_img0, train_target, train_pred_palm,train_uvd_centre= load_data(save_dir,dataset=dataset)
        n_train_batches=int(train_img0.shape[0]/batch_size)
        train_idx=range(train_img0.shape[0])
        offset=numpy.empty_like(train_target)
        for minibatch_index in range(n_train_batches):
        # for minibatch_index in range(3):
            batch_idx = train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x0,x1,x2,y=data_augmentation.get_img_for_palm_holi(r0=train_img0[batch_idx],
                                                                gr_uvd=train_target[batch_idx],pred_uvd=train_pred_palm[batch_idx],
                                                                uvd_hand_center=train_uvd_centre[batch_idx],if_aug=False)
            offset[batch_idx] = model.predict_on_batch(x={'input0':x0,'input1':x1,'input2':x2}).reshape(-1,6,3)
            #
            # numpy.save("%s/hier/result/%s_x0_%s"%(save_dir,dataset,version),[x0])
            # numpy.save("%s/hier/result/%s_offset_%s"%(save_dir,dataset,version),[y,offset])
        normuvd = get_err.get_normuvd_from_offset(offset=offset,previous_pred_uvd=train_pred_palm,uvd_hand_center=train_uvd_centre)
        print('shape of pred and ori',y.shape,normuvd.shape)
        normuvd.shape=(normuvd.shape[0],len(palm_idx),3)
        numpy.save("%s/hier/result/%s_normuvd_%s"%(save_dir,dataset,version),normuvd)
        dataset='test'
        normuvd = numpy.load("%s/hier/result/%s_normuvd_%s.npy"%(save_dir,dataset,version))
        path='%s/source/%s_crop_norm_vassi.h5'%(save_dir,dataset)
        xyz_pred, xyz_gt, xyz_err = get_err.get_err_from_normuvd(path=path,dataset=dataset,normuvd=normuvd,jnt_idx=palm_idx,setname='mega')
def show():
    dataset='test'
    save_dir = 'F:/HuaweiProj/data/mega'
    x0=numpy.load("%s/hier/result/%s_x0_%s.npy"%(save_dir,dataset,version))[0]
    tmp = numpy.load("%s/hier/result/%s_offset_%s.npy"%(save_dir,dataset,version))

    y=tmp[0]
    offset = tmp[1]
    print(y.shape,offset.shape)
    y.shape=(y.shape[0],6,3)
    for i in range(offset.shape[0]):
        plt.imshow(x0[i,:,:,0],'gray')
        plt.scatter(y[i,:,0]*96+48,y[i,:,1]*96+48,c='r')
        plt.scatter(offset[i,:,0]*96+48,offset[i,:,1]*96+48,c='b')
        plt.show()

if __name__ == '__main__':
    # train()
    train()
    # predict()
    # show()
