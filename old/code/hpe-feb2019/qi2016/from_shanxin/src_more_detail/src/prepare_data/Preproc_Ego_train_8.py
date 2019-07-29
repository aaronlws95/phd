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
            new_xyz_jnt_gt.append(xyz_jnt_gt[i])
            new_uvd_jnt_gt.append(uvd_jnt_gt[i])
            new_file_name.append(file_name[i])

            # mean_z = numpy.mean(points0[:,2])#mean_z is th avarage depth of the hand point could before moving the hand to the reference depth
            mean_z = xyz_hand_jnts_gt[center_joint_idx,2]
            points0[:,2] += ref_z-mean_z
            xyz_hand_jnts_gt[:,2] += ref_z-mean_z
            hand_points_uvd = convert.xyz2uvd(xyz=points0)

            hand_jnt_gt_uvd= convert.xyz2uvd(xyz=xyz_hand_jnts_gt)

            mean_u = hand_jnt_gt_uvd[center_joint_idx,0]
            mean_v = hand_jnt_gt_uvd[center_joint_idx,1]

            hand_points_uvd[:,0] = hand_points_uvd[:,0] - mean_u+ bbox_uvd[0,0]/2
            hand_points_uvd[ numpy.where(hand_points_uvd[:,0]>=bbox_uvd[0,0]),0]=bbox_uvd[0,0]-1
            hand_points_uvd[ numpy.where(hand_points_uvd[:,0]<0),0]=0

            hand_jnt_gt_uvd[:,0] = ( hand_jnt_gt_uvd[:,0] - mean_u + bbox_uvd[0,0]/2 ) / bbox_uvd[0,0]
            hand_points_uvd[ numpy.where(hand_jnt_gt_uvd[:,0]>1),0]=1
            hand_points_uvd[ numpy.where(hand_jnt_gt_uvd[:,0]<0),0]=0

            hand_points_uvd[:,1] = hand_points_uvd[:,1] - mean_v+bbox_uvd[0,1]/2
            hand_points_uvd[ numpy.where(hand_points_uvd[:,1]>=bbox_uvd[0,1]),1]=bbox_uvd[0,1]-1
            hand_points_uvd[ numpy.where(hand_points_uvd[:,1]<0),1]=0

            hand_jnt_gt_uvd[:,1] =( hand_jnt_gt_uvd[:,1] - mean_v+bbox_uvd[0,1]/2 ) / bbox_uvd[0,1]
            hand_jnt_gt_uvd[ numpy.where(hand_jnt_gt_uvd[:,1]>1),1]=1
            hand_jnt_gt_uvd[ numpy.where(hand_jnt_gt_uvd[:,1]<0),1]=0


            hand_points_uvd[:,2] = (hand_points_uvd[:,2] - ref_z +bbsize/2)/bbsize
            hand_jnt_gt_uvd[:,2] = (hand_jnt_gt_uvd[:,2] - ref_z +bbsize/2)/bbsize

            new_hand = numpy.ones((int(bbox_uvd[0,1]),int(bbox_uvd[0,0])),dtype='float32')
            # print bbox_uvd
            new_hand[numpy.asarray(numpy.floor(numpy.squeeze(hand_points_uvd[:,1])),dtype='int16'),
                     numpy.asarray(numpy.floor(numpy.squeeze(hand_points_uvd[:,0])),dtype='int16')]\
                =hand_points_uvd[:,2]




            hand_center_uvd.append([mean_u,mean_v, mean_z])
            uvd_jnt_gt_norm.append(hand_jnt_gt_uvd)

            r0_tmp= interplt.zoom(new_hand, img_size/bbox_uvd[0,0],order=1, mode='nearest',prefilter=True)
            r1_tmp= interplt.zoom(new_hand, img_size/bbox_uvd[0,0]/2,order=1, mode='nearest',prefilter=True)
            r2_tmp=interplt.zoom(new_hand, img_size/bbox_uvd[0,0]/4,order=1, mode='nearest',prefilter=True)
            r0.append(r0_tmp)
            r1.append(r1_tmp)
            r2.append(r2_tmp)


    print 'num of wrong anotation',len(xyz_jnt_gt)-len(new_xyz_jnt_gt)
    print 'num of valid samples',len(new_xyz_jnt_gt)
    f = h5py.File(save_path, 'w')
    f.create_dataset('r0', data=r0)
    f.create_dataset('r1', data=r1)
    f.create_dataset('r2', data=r2)
    f.create_dataset('uvd_jnt_gt', data=new_uvd_jnt_gt)
    f.create_dataset('xyz_jnt_gt', data=new_xyz_jnt_gt)
    f.create_dataset('uvd_jnt_gt_norm', data=uvd_jnt_gt_norm)
    f.create_dataset('hand_center_uvd', data=hand_center_uvd)

    f.create_dataset('resxy', data=[resx,resy])
    f.create_dataset('ref_z', data=ref_z)
    f.create_dataset('bbsize', data=bbsize)
    f.close()


if __name__ == '__main__':
    #

    jointlocations = 'trainingloc_All_EGO_CNN_Caner'
    save_path = '../../data/ego/Ego_norm_hand_uvd_rootmid_%s.h5' % jointlocations

    norm_hand(save_path=save_path,
    dataset_dir = '/media/Data/shanxin/megahand/', jointlocations = jointlocations)

