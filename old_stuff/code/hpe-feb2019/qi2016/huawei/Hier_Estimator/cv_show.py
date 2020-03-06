__author__ = 'QiYE'

import numpy

import h5py
import cv2
import os
from PIL import Image

import time
img_dir = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/MegaEgo/'

colors = numpy.array([[0.,0,0],
              [1.0,.0,0],
              [0.8,.0,0],
              [0.6,0,0],
              [0.4,0,0],

              [0,1,0],
              [0,0.8,0],
              [0,0.6,0],
              [0,0.4,0],

              [0,0,1],
              [0,0,0.8],
              [0,0,0.6],
              [0,0,0.4],

              [1,1,0],
              [1,0.8,0],
              [1.,0.6,0],
              [1,0.4,0],

              [1,0,1],
              [1.,0,0.8],
              [1,0,0.6],
              [1,0,0.4],
              ]).reshape(21,3)*255
# colors=numpy.array(colors*255,dtype='uint8')
print(colors.shape)


def show_2D_hand_skeleton(imgcopy,uvd_pred):
    print(uvd_pred.shape)


    ratio_size=int(1500.0/numpy.mean(uvd_pred[0,:,2]))

    for k in [1,5,9,13,17]:
        print(tuple(colors[k]))
        cv2.line(imgcopy,tuple(uvd_pred[0,0,0:2]), tuple(uvd_pred[0,k,0:2]), tuple(colors[k]), ratio_size)
        cv2.line(imgcopy,tuple(uvd_pred[0,k,0:2]), tuple(uvd_pred[0,k+1,0:2]), tuple(colors[k+1]), ratio_size)
        cv2.line(imgcopy,tuple(uvd_pred[0,k+1,0:2]), tuple(uvd_pred[0,k+2,0:2]), tuple(colors[k+2]), ratio_size)
        cv2.line(imgcopy,tuple(uvd_pred[0,k+2,0:2]), tuple(uvd_pred[0,k+3,0:2]), tuple(colors[k+3]), ratio_size)
    ratio_size=int(3000.0/numpy.mean(uvd_pred[0,:,2]))
    for j in range(uvd_pred.shape[1]):
        cv2.circle(imgcopy,(int(uvd_pred[0,j,0]),int(uvd_pred[0,j,1])), ratio_size, tuple(colors[j]), -1)
    return imgcopy

def main():



    dataset='test'
    f = h5py.File('F:/HuaweiProj/data/mega/source/%s_crop_norm_v1.h5'%(dataset), 'r')
    # f = h5py.File('F:/HuaweiProj/data/mega/source/%s_crop_norm_v1.h5'%(dataset), 'r')
    new_file_names = f['new_file_names'][...]
    gt_xyz= f['xyz_gt'][...]
    uvd_pred= numpy.array(f['uvd_gt'][...],dtype='int32')
    f.close()

    for i in range(0,new_file_names.shape[0],1):
    # for i in numpy.random.randint(0,new_file_names.shape[0],100):
        cur_frame=new_file_names[i]
        depth = Image.open("%s%s.png"%(img_dir,cur_frame))
        depth = numpy.asarray(depth, dtype='uint16')
        imgcopy=depth.copy()
        # msk = numpy.logical_and(2000 > imgcopy, imgcopy > 0)
        # msk2 = numpy.logical_or(imgcopy == 0, imgcopy == 1000)
        min = imgcopy.min()
        max = imgcopy.max()
        imgcopy = (imgcopy - min) / (max - min) * 255.
        imgcopy = imgcopy.astype('uint8')
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)
        # poseimg=depth.copy()
        # poseimg = numpy.zeros_like(depth)
        imgcopy=show_2D_hand_skeleton(imgcopy,numpy.array(uvd_pred[[i]],dtype='int32'))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', imgcopy)
        cv2.waitKey(1)

if __name__=='__main__':

    main()
