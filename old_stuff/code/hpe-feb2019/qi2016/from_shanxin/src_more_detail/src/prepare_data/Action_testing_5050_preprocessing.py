__author__ = 'Shanxin'
__author__ = 'Shanxin'

import numpy
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage.interpolation  as interplt
from matplotlib import pylab
import h5py
from src.utils import convert
import csv
from mpl_toolkits.mplot3d import Axes3D
import scipy.io



def norm_hand(save_path,dataset_dir,jointlocations):
    print save_path
    u0= 315.944855
    v0= 245.287079
    img_size = 96.0
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]


    resx = 640
    resy = 480
    ref_z = 1000.0
    bbsize = 260.0

    center_joint_idx = 9

    xyz_jnt_gt=[]
    file_name = []

    with open('%s%s.txt'%(dataset_dir,jointlocations), 'rb') as f:
        for line in f:
            part = line.split('\t')

            file_name.append(part[0].replace('\\', '/'))
            xyz_jnt_gt.append(part[1:64])

    f.close()

    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float32')

    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt =convert.xyz2uvd(xyz_jnt_gt)
    new_xyz_jnt_gt =convert.uvd2xyz(uvd_jnt_gt)
    err =numpy.sqrt(numpy.sum((new_xyz_jnt_gt-xyz_jnt_gt)**2,axis=-1))
    print 'mean err ', numpy.mean(numpy.mean(err))
    bb = numpy.array([(bbsize,bbsize,ref_z)])


    bbox_uvd = convert.xyz2uvd(xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - u0)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - v0)
    new_file_name=[]
    new_xyz_jnt_gt=[]
    new_uvd_jnt_gt=[]
    uvd_jnt_gt_norm=[]
    hand_center_uvd=[]
    r0=[]
    r1=[]
    r2=[]
    # for i in xrange(0,10,1):
    for i in xrange(0,xyz_jnt_gt.shape[0],1):
    # for i in xrange(0, 1000, 1):
        if i%500 ==0:
            print i

        depth = Image.open('%s%s' % (dataset_dir,file_name[i]))
        depth = numpy.asarray(depth, dtype='uint16')
        # print depth.shape

        uvd = convert.convert_depth_to_uvd(depth).astype(dtype='float32')

        xyz = convert.uvd2xyz(uvd=uvd)
        points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)
        xyz_hand_jnts_gt = numpy.squeeze(xyz_jnt_gt[i]).copy()

        # % Collect the points within the AABBOX of the hand
        axis_bounds = numpy.array([numpy.min(xyz_hand_jnts_gt[:, 0]), numpy.max(xyz_hand_jnts_gt[:, 0]),
                                   numpy.min(xyz_hand_jnts_gt[:, 1]), numpy.max(xyz_hand_jnts_gt[:, 1]),
                                   numpy.min(xyz_hand_jnts_gt[:, 2]), numpy.max(xyz_hand_jnts_gt[:, 2])])

        mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
        axis_bounds[mask] -= 20
        axis_bounds[~mask] += 20
        mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
        mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
        mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
        inputs = mask1 & mask2 & mask3


        points0 = points[inputs]

        if len(points0) == 0:
            print i, '*******************wrong anotation******************',file_name[i]
        else:
            # record the file name
            new_file_name.append(file_name[i])

    scipy.io.savemat('%s' % (save_path), {'filenames': new_file_name})




if __name__ == '__main__':
    #

    jointlocations = 'training_50_50_Mar_08_2017'
    save_path = '../../data/action/Training_filename.mat'

    norm_hand(save_path=save_path,
    dataset_dir = '/media/Data/shanxin/megahand/', jointlocations = jointlocations)

