__author__ = 'QiYE'

import sys

import h5py
import numpy
import scipy.io
from src.utils import constants

from utils_online import recur_derot, crop_bw_ego_conv_patch, err_in_ori_xyz_oneframe, norm_01, get_rot, rot_img, crop_patch,offset_to_abs
import cv2
import File_Name_HDR
import set_model_parameter
import matplotlib.pyplot as plt


batch_size=1
dataset='test'
source_name=File_Name_HDR.icvl_source_name
frame_idx=0
offset_depth_range=0.4

class Hier_SA:
    def __init__(self):
        self.jntmodels=[]
        setname='icvl'
        self.jntmodels+=set_model_parameter.set_fig_model(setname)
        print len(self.jntmodels)


    def predict(self, img0, img1, img2):


        path = 'D:\icvl_tmp\icvl_iter_whlimg_derot.h5'
        f = h5py.File(path,'r')
        r0 = f['r0'][...][frame_idx]
        r1 = f['r1'][...][frame_idx]
        r2 = f['r2'][...][frame_idx]
        joint_label_uvd = f['gr_uvd_derot'][...]
        f.close()

        uvd_offset=numpy.empty((1,5,3),dtype='float32')
        i=0
        center_uvd = joint_label_uvd[0,i+1]
        print center_uvd
        tmp0, tmp1, tmp2 = crop_patch(center_uvd.reshape(1, 3),  r0.reshape(1,1,96,96),  r1.reshape(1,1,48,48),  r2.reshape(1,1,24,24),
                                      patch_size=40,  patch_pad_width=4, hand_width=96,  pad_width=0)

        uvd_offset[:,  i, :] = self.jntmodels[0](tmp0.reshape(1,1,48,48),  tmp1.reshape(1,1,24,24),  tmp2.reshape(1,1,14,14),  numpy.cast['int32'](0))
        plt.imshow(tmp0[0], 'gray')
        plt.scatter(uvd_offset[:,i, 0]*40, uvd_offset[:,i, 1]*40)
        plt.show()


        return uvd_pred





if __name__ == '__main__':
    """change the NUM_JNTS in src/constants.py to 6"""
    """!!!!!!When cropping pathces make sure the kernel in constants is the same with model used!!!!!!!!"""
    setname='icvl'
    dataset_path_prefix=constants.Data_Path
    '''get patches in new veiwpoint by the model of the bw inital stage'''

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix, setname, dataset, setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix, setname, dataset, setname))
    roixy = keypoints['roixy']

    path = '%sdata/%s/source/%s%s.h5'%(dataset_path_prefix, setname, dataset, source_name)
    f = h5py.File(path, 'r')
    r0=f['r0'][...]
    r1=f['r1'][...]
    r2=f['r2'][...]
    uvd_gr = f['joint_label_uvd'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    f.close()
    
    a = Hier_SA()


    uvd_pred=a.predict(r0[frame_idx],r1[frame_idx],r2[frame_idx])
    jnt_idx=range(0,21,1)
    xyz_pred, err = err_in_ori_xyz_oneframe(uvd_pred, uvd_gr[frame_idx], xyz_true[frame_idx],
                                       roixy[frame_idx], rect_d1d2w[frame_idx], depth_dmin_dmax[frame_idx], orig_pad_border, jnt_idx=jnt_idx)


    print 'current result', xyz_pred
    print 'previous result'
    for i in xrange(21):
        jnt_xyz = scipy.io.loadmat('D:/icvl_tmp/jnt%d_xyz.mat'%i)['jnt']
        print jnt_xyz[frame_idx]
