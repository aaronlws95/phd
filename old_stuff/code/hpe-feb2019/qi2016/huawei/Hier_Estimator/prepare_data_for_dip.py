__author__ = 'QiYE'

import numpy
import h5py
import matplotlib.pyplot as plt
import cv2
from ..utils import math
from skimage.transform import resize

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
from ..utils import get_err,hand_utils,data_augmentation
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
palmjnt=0

    save_dir='/media/Data/Qi/data'
    version = 'palm_s0_rot_scaleker32_lr0.000030'
palm_idx =[0,1,5,9,13,17]

def crop_patch_allpalmjnt(r0,pred_uv,r0_patch_size):
    border=r0_patch_size
    jnt_uv=pred_uv.copy()
    jnt_uv[numpy.where(pred_uv>r0.shape[0]+r0_patch_size/2)]=r0.shape[0]+r0_patch_size/2
    jnt_uv[numpy.where(pred_uv<-r0_patch_size/2)]=-r0_patch_size/2
    jnt_uv=numpy.array(numpy.round(jnt_uv),dtype='uint16')
    r0_patch_half_size = int(r0_patch_size/2)

    r0_tmp = numpy.lib.pad(r0, ((border,border),(border,border)), 'constant',constant_values=1)
    p1 = numpy.empty((pred_uv.shape[0],r0_patch_size,r0_patch_size),dtype='float32')

    for i in range(pred_uv.shape[0]):
        vmin = jnt_uv[i,1]-r0_patch_half_size+border
        vmax = jnt_uv[i,1]+r0_patch_half_size+border
        umin= jnt_uv[i,0]-r0_patch_half_size+border
        umax =jnt_uv[i,0]+r0_patch_half_size+border
        p1[i] = r0_tmp[vmin:vmax,umin:umax]

    return p1

def load_test_data(save_dir):

    f = h5py.File('%s/source/test_crop_norm.h5'%save_dir, 'r')
    test_x0 = f['img0'][...]
    test_y= f['uvd_norm_gt'][...][:,palm_idx,:]
    f.close()
    print('testset loaded',test_x0.shape,test_y.shape)
    normuvd = numpy.load("%s/hier/result/test_normuvd_palm_s0_rot_scaleker32_lr0.000030.npy"%(save_dir))
    numimg=normuvd.shape[0]

    return numpy.expand_dims(test_x0[:numimg],axis=-1),test_y[:numimg],normuvd[:numimg]


def get_hand_part_for_palm(r0,gr_uvd,pred_uvd):
    new_r0=r0[:,:,:,0].copy()


    num_frame=gr_uvd.shape[0]

    rot_angle = math.get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])

    crop0=numpy.empty((num_frame,6,48,48,1),dtype='float32')
    crop1 = numpy.empty((num_frame,6,24,24,1),dtype='float32')
    target = numpy.empty((num_frame,6,3),dtype='float32')
    target[:,:,2]=gr_uvd[:,:,2]-pred_uvd[:,:,2]

    for i in range(0,gr_uvd.shape[0],1):
        print(i)
        M = cv2.getRotationMatrix2D((48,48),rot_angle[i],1)
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(gr_uvd.shape[1]):
            gr_uvd[i,j,0:2] = numpy.dot(M,numpy.array([gr_uvd[i,j,0]*96+48,gr_uvd[i,j,1]*96+48,1]))
            pred_uvd[i,j,0:2] = numpy.dot(M,numpy.array([pred_uvd[i,j,0]*96+48,pred_uvd[i,j,1]*96+48,1]))

        crop0[i,:,:,:,0]=crop_patch_allpalmjnt(r0=new_r0[i],pred_uv=pred_uvd[i,:,0:2],r0_patch_size=48)
        for j in range(gr_uvd.shape[1]):
            crop1[i,j,:,:,0] = resize(crop0[i,j,:,:,0], (24,24), order=3,preserve_range=True)

        target[i,:,0:2]=(gr_uvd[i,:,0:2]-pred_uvd[i,:,0:2])/96.0

        # for j in range(gr_uvd.shape[1]):
        #     fig = plt.figure()
        #     ax =fig.add_subplot(221)
        #     ax.imshow(new_r0[i],'gray')
        #     ax.scatter(gr_uvd[i,:,0],gr_uvd[i,:,1],c='r')
        #     ax.scatter(pred_uvd[i,:,0],pred_uvd[i,:,1],c='b')
        #
        #
        #     ax =fig.add_subplot(222)
        #     ax.imshow(crop1[i,j,:,:,0],'gray')
        #
        #     ax =fig.add_subplot(223)
        #     ax.imshow(crop0[i,j,:,:,0],'gray')
        #     ax.scatter(target[i,j,0]*48+24,target[i,j,1]*48+24,c='r')
        #
        #     ax =fig.add_subplot(224)
        #     ax.imshow(r0[i,:,:,0],'gray')
        #     # plt.scatter(pred_uvd[i,:,0]*96+48,pred_uvd[i,:,1]*96+48,c='b')
        #     # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        #     plt.show()
    return crop0,crop1,target
    # hand_utils.show_two_palm_skeleton(gr_uvd[i],pred_uvd[i])


def predict():

    test_loss = numpy.load("%s/hier/model/history_%s.npy"%(save_dir,version))[-1]
    print(numpy.min(test_loss))
    json_file = open("%s/hier/model/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')

    r0,gr_uvd,pred_uvd = load_test_data(save_dir,dataset)
    crop0,crop1,target = get_hand_part_for_palm(r0,gr_uvd,pred_uvd)
    batch_size=128
    train_img0,train_target,train_pred_palm = load_test_data(save_dir,dataset)

    offset = numpy.empty_like(train_target)
    for jnt in palm_idx:

        # load weights into new model
        model.load_weights("%s/hier/model/weight_%s"%(save_dir,version))
        dataset='test'

        n_train_batches=int(train_img0.shape[0]/batch_size)
        train_idx=range(train_img0.shape[0])


        for minibatch_index in range(n_train_batches):
            batch_idx = train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x0,x1,y=data_augmentation.get_hand_part_for_palm(r0=train_img0[batch_idx],
                                                                gr_uvd=train_target[batch_idx],pred_uvd=train_pred_palm[batch_idx],jnt=palmjnt,if_aug=False)
            offset[batch_idx,jnt] = model.predict_on_batch(x={'input0':x0,'input1':x1})
        numpy.save("%s/hier/result/%s_offset_%s"%(save_dir,dataset,version),offset)

    normuvd = get_err.get_normuvd_from_offset(offset=offset,previous_pred_uvd=train_pred_palm)
    print('shape of pred and ori',y.shape,normuvd.shape)
    normuvd.shape=(normuvd.shape[0],len(palm_idx),3)
    numpy.save("%s/hier/result/%s_normuvd_%s"%(save_dir,dataset,version),normuvd)

    # normuvd = numpy.load("%s/hier/result/normuvd_%s.npy"%(save_dir,version))
    xyz_pred, xyz_gt, xyz_err = get_err.get_err_from_normuvd(base_dir=save_dir,dataset=dataset,normuvd=normuvd,jnt_idx=palm_idx,setname='mega')


if __name__=='__main__':
    predict()

