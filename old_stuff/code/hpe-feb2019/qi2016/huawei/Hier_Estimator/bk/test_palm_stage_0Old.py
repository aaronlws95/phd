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


palmjnt=0

save_dir='/media/Data/Qi/data'
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

def load_test_data(save_dir,dataset):

    f = h5py.File('%s/source/%s_crop_norm.h5'%(save_dir,dataset), 'r')
    valid_idx =f['valid_idx'][...]
    test_x0 = f['img0'][...][valid_idx]
    test_y= f['uvd_norm_gt'][...][:,palm_idx,:][valid_idx]
    f.close()

    print('testset loaded',test_x0.shape,test_y.shape)
    normuvd = numpy.load("%s/hier/result/%s_normuvd_palm_s0_rot_scaleker32_lr0.000030.npy"%(save_dir,dataset))
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
def load_test_data_for_pred(save_dir,dataset):


    f = h5py.File('%s/source/%s_crop_norm_v1.h5'%(dataset,save_dir), 'r')
    valid_idx =f['valid_idx'][...]

    test_x0 = f['img0'][...][valid_idx]
    test_x1 = f['img1'][...][valid_idx]
    test_x2 = f['img2'][...][valid_idx]
    test_y= f['uvd_norm_gt'][...][:,palm_idx,:].reshape(test_x0.shape[0],-1)
    test_y=test_y[valid_idx]
    f.close()
    print('testset loaded',test_x0.shape,test_y.shape)

    return numpy.expand_dims(test_x0,axis=-1),numpy.expand_dims(test_x1,axis=-1),numpy.expand_dims(test_x2,axis=-1),test_y
def predict(dataset):
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

    x0,x1,x2,y = load_test_data_for_pred(save_dir,dataset)

    loaded_model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')
    normuvd = loaded_model.predict(x={'input0':x0,'input1':x1,'input2':x2},batch_size=128)
    print('shape of pred and ori',y.shape,normuvd.shape)
    normuvd.shape=(normuvd.shape[0],len(palm_idx),3)
    numpy.save("%s/hier/result/%s_normuvd_%s"%(save_dir,dataset,version),normuvd)
    # normuvd = numpy.load("%s/hier/result/normuvd_%s.npy"%(save_dir,version))
    xyz_pred, xyz_gt, xyz_err = get_err.get_err_from_normuvd(base_dir=save_dir,dataset=dataset,normuvd=normuvd,jnt_idx=palm_idx,setname='mega')



if __name__=='__main__':
    dataset='test'
    predict(dataset)

