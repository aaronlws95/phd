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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
setname='mega'
#
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.2):
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
num_kern=[48,96]

lr=0.0001
cur_finger=1
cur_jnt_idx=[cur_finger*4+1+1]
jnt_in_prev_layer=[cur_finger+1]
version = 'pip_s0_finger%d_smalljiter_ker%d_lr%f'%(cur_finger,num_kern[0],lr)
# save_dir = 'F:/HuaweiProj/data/mega'
save_dir='/media/Data/Qi/data'
batch_size=128


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses=[]

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        numpy.save('%s/detector/loss_history_%s'%(save_dir,version),[self.losses,self.val_losses])


def load_data(save_dir,dataset,cur_jnt_idx,jnt_in_prev_layer):

    f = h5py.File('%s/source/%s_crop_norm_vassi.h5'%(save_dir,dataset), 'r')
    train_x0 = f['img0'][...]
    train_y= f['uvd_norm_gt'][...][:,cur_jnt_idx,:]
    f.close()

    normuvd = numpy.load("%s/hier/result/%s_normuvd_tmp_vass_palm_s0_rot_scale_ker32_lr0.000100.npy"%(save_dir,dataset))
    jnt_uvd_in_prev_layer=normuvd[:,jnt_in_prev_layer,:]
    print(dataset, ' train_x0.shape,train_y.shape,normuvd.shape',train_x0.shape,train_y.shape,normuvd.shape,jnt_uvd_in_prev_layer.shape)

    return numpy.expand_dims(train_x0,axis=-1), train_y,normuvd,jnt_uvd_in_prev_layer


def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = multi_resolution.get_multi_reso_for_finger(img_rows=48,img_cols=48,
                                             num_kern=num_kern,num_f1=1024,num_f2=1024,cnn_out_dim=3)
    model.compile(optimizer=Adam(lr=lr),loss='mean_squared_error')
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s/hier/%s.json"%(save_dir,version), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("%s/hier/weight_%s_bf"%(save_dir,version))

    train_x0, train_y,train_pred_palm,train_jnt_uvd_in_prev_layer= load_data(save_dir,dataset='train',cur_jnt_idx=cur_jnt_idx,jnt_in_prev_layer=jnt_in_prev_layer)
    test_x0,test_target,test_pred_palm,test_jnt_uvd_in_prev_layer = load_data(save_dir,dataset='test',cur_jnt_idx=cur_jnt_idx,jnt_in_prev_layer=jnt_in_prev_layer)
    trainer.fit_pip_stage_0(batch_size=batch_size,n_epochs=1000,model=model,
          train_img0=train_x0,train_target=train_y,train_pred_palm=train_pred_palm,train_jnt_uvd_in_prev_layer=train_jnt_uvd_in_prev_layer,
          test_img0=test_x0,test_target=test_target,test_pred_palm=test_pred_palm,test_jnt_uvd_in_prev_layer=test_jnt_uvd_in_prev_layer,
          model_filepath='%s/hier/weight_%s'%(save_dir,version),
          history_filepath='%s/hier/history_%s'%(save_dir,version),if_aug=True,aug_trans=0.02,aug_rot=15)

def predict():
    scale=1.8
    cur_finger=4
    cur_jnt_idx=[cur_finger*4+1+1]
    jnt_in_prev_layer=[cur_finger+1]
    version='pip_s0_finger%d_smalljiter_ker48_lr0.000100'%cur_finger
    tmp = numpy.load("%s/hier/history_%s_epoch.npy"%(save_dir,version))
    test_loss=tmp[1]
    train_loss=tmp[0]
    loc=numpy.argmin(test_loss)
    print(version,test_loss[loc],train_loss[loc])
    json_file = open("%s/hier/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')
    model.load_weights("%s/hier/weight_%s"%(save_dir,version))



    batch_size=128
    for dataset in ['test','train']:
        train_img0, train_target,train_pred_palm,train_jnt_uvd_in_prev_layer= load_data(save_dir,dataset=dataset,cur_jnt_idx=cur_jnt_idx, jnt_in_prev_layer=jnt_in_prev_layer)

        offset = numpy.empty_like(train_target)

        n_train_batches=int(train_img0.shape[0]/batch_size)
        train_idx=range(train_img0.shape[0])

        for minibatch_index in range(n_train_batches):
            batch_idx = train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x0,x1,y=data_augmentation.get_crop_for_finger_part_s0(r0=train_img0[batch_idx],
                                                                gr_uvd=train_target[batch_idx],
                                                                pred_uvd=train_pred_palm[batch_idx],
                                                                jnt_uvd_in_prev_layer=train_jnt_uvd_in_prev_layer[batch_idx],if_aug=False,scale=scale)
            tmp = model.predict_on_batch(x={'input0':x0,'input1':x1})
            offset[batch_idx] =tmp.reshape(batch_size,-1,3)

            # numpy.save("%s/hier/result/%s_offset_%s"%(save_dir,dataset,version),[y,tmp])
        normuvd = get_err.get_normuvd_from_offset(offset=offset,pred_palm=train_pred_palm,
                                                          jnt_uvd_in_prev_layer=train_jnt_uvd_in_prev_layer,scale=scale)
        print('shape of pred and ori',train_target.shape,normuvd.shape)
        normuvd.shape=(normuvd.shape[0],len(cur_jnt_idx),3)
        numpy.save("%s/hier/result/%s_normuvd_%s"%(save_dir,dataset,version),normuvd)

        # normuvd = numpy.load("%s/hier/result/normuvd_%s.npy"%(save_dir,version))
        path='%s/source/%s_crop_norm_vassi.h5'%(save_dir,dataset)
        xyz_pred, xyz_gt, xyz_err = get_err.get_err_from_normuvd(path=path,dataset=dataset,normuvd=normuvd,jnt_idx=cur_jnt_idx,setname='mega')


if __name__ == '__main__':
    # train()
    # train()
    predict()
    # train()
