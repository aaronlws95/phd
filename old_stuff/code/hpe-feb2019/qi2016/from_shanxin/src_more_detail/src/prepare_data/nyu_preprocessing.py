__author__ = 'QiYE'


import numpy
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage.interpolation  as interplt
from matplotlib import pylab
import h5py
from src.utils import xyz_uvd
import csv
from mpl_toolkits.mplot3d import Axes3D
import scipy.io



def norm_hand(save_path,setname,
              label_path,
     dataset,
     num_smpl,
    fram_prefix,
    fram_pstfix ,
    dataset_dir ):

    img_size = 96
    our_index = [35,5,3,1,0,11,9,7,6,17,15,13,12,23,21,19,18,29,27,25,24]
    roixy = numpy.zeros((num_smpl,4),dtype='float32')

    resx = 640
    resy = 480
    u0= 320
    v0= 240
    ref_z = 1300.0
    bbsize = 260.0
    center_joint_idx = 9

    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float32')

    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt =xyz_uvd.xyz2uvd(xyz_jnt_gt)
    new_xyz_jnt_gt =xyz_uvd.uvd2xyz(uvd_jnt_gt)
    err =numpy.sqrt(numpy.sum((new_xyz_jnt_gt-xyz_jnt_gt)**2,axis=-1))
    print 'mean err ', numpy.mean(numpy.mean(err))
    bb = numpy.array([(bbsize,bbsize,ref_z)])


    bbox_uvd = xyz_uvd.xyz2uvd(xyz=bb)
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


    for i in xrange(0,xyz_jnt_gt.shape[0],1):
        print i
        filename_prefix = "%07d" % (i+1)
        depth = Image.open('%s%s_%s%s.png' % (dataset_dir, fram_prefix, filename_prefix,fram_pstfix))
        depth = numpy.asarray(depth, dtype='uint16')
        depth = depth[:, :, 2]+numpy.left_shift(depth[:, :, 1], 8)


        uvd = xyz_uvd.convert_depth_to_uvd(depth).astype(dtype='float32')

        xyz = xyz_uvd.uvd2xyz(uvd=uvd)
        points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)
        xyz_hand_jnts_gt = numpy.squeeze(xyz_jnt_gt[i]).copy()
        # # #
        # tmp_uvd=uvd_jnt_gt[i]
        # tmp_uvd[:,0]-=roixy[i,0]
        # tmp_uvd[:,1]-=roixy[i,2]
        # plt.imshow(depth,'gray')
        # plt.scatter(tmp_uvd[:,0],tmp_uvd[:,1])
        # plt.show()
        # % Collect the points within the AABBOX of the hand
        axis_bounds = numpy.array([numpy.min(xyz_hand_jnts_gt[:, 0]), numpy.max(xyz_hand_jnts_gt[:, 0]),
                                   numpy.min(xyz_hand_jnts_gt[:, 1]), numpy.max(xyz_hand_jnts_gt[:, 1]),
                                   numpy.min(xyz_hand_jnts_gt[:, 2]), numpy.max(xyz_hand_jnts_gt[:, 2])])
        # print 'hand bbox in xyz'
        # print numpy.max(xyz_hand_jnts_gt[:, 2]), numpy.min(xyz_hand_jnts_gt[:, 2]),numpy.max(xyz_hand_jnts_gt[:, 2])- numpy.min(xyz_hand_jnts_gt[:, 2])
        # print numpy.max(xyz_hand_jnts_gt[:, 1]), numpy.min(xyz_hand_jnts_gt[:, 1]),numpy.max(xyz_hand_jnts_gt[:, 1])- numpy.min(xyz_hand_jnts_gt[:, 1])
        # print numpy.max(xyz_hand_jnts_gt[:, 0]), numpy.min(xyz_hand_jnts_gt[:, 0]),numpy.max(xyz_hand_jnts_gt[:, 0])-numpy.min(xyz_hand_jnts_gt[:, 0])

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
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
            # # ax.scatter3D(points[:, 2], points[:, 0], points[:, 1], s=1.5, marker='.')
            # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, marker='.')
            # # ax.plot([xyz_hand_jnts_gt[0,2],direc[2]*100+xyz_hand_jnts_gt[0,2]],
            # #         [xyz_hand_jnts_gt[0,1],direc[1]*100+xyz_hand_jnts_gt[0,1]],
            # #         [xyz_hand_jnts_gt[0,0],direc[0]*100+xyz_hand_jnts_gt[0,0]])
            # #
            # # ax.plot([xyz_hand_jnts_gt[0,2],direc[2]*100+xyz_hand_jnts_gt[0,2]],
            # #             [xyz_hand_jnts_gt[0,1],direc[1]*100+xyz_hand_jnts_gt[0,1]],
            # #             [xyz_hand_jnts_gt[0,0],direc[0]*100+xyz_hand_jnts_gt[0,0]])
            # ax.scatter3D(xyz_hand_jnts_gt[:,2], xyz_hand_jnts_gt[:,0], xyz_hand_jnts_gt[:,1], c = 'r',marker='o',s=20)
            # # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5, marker='.')
            # # ax.scatter3D(hand_points[:,2], hand_points[:,0], hand_points[:,1], c = 'r',marker='o',s=20)
            # #ax.plot_surface(points0[:, 2],points0[:, 0], points0[:, 1],rstride=1, cstride=1,color='b')
            # ax.set_xlim(axis_bounds[5], axis_bounds[4])
            # ax.set_ylim(axis_bounds[0], axis_bounds[1])
            # ax.set_zlim(axis_bounds[2], axis_bounds[3])
            # ax.azim =0
            # ax.elev = 180
            # plt.show()

            # mean_z = numpy.mean(points0[:,2])#mean_z is th avarage depth of the hand point could before moving the hand to the reference depth
            mean_z = xyz_hand_jnts_gt[center_joint_idx,2]
            points0[:,2] += ref_z-mean_z
            xyz_hand_jnts_gt[:,2] += ref_z-mean_z
            hand_points_uvd = convert.xyz2uvd(xyz=points0)

            hand_jnt_gt_uvd= convert.xyz2uvd(xyz=xyz_hand_jnts_gt)



            # print 'hand bbox in uvd at dist 500'
            # print bbox_uvd
            # mean_u =numpy.mean(hand_points_uvd[:,0])#mean_u is th avarage u axis width of the hand image after moving the hand to the reference depth
            # mean_v = numpy.mean(hand_points_uvd[:,1])#mean_v is th avarage v axis height of the hand image after moving the hand to the reference depth
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

            new_hand = numpy.ones((bbox_uvd[0,1],bbox_uvd[0,0]),dtype='float32')
            # print bbox_uvd
            new_hand[numpy.asarray(numpy.floor(numpy.squeeze(hand_points_uvd[:,1])),dtype='int16'),
                     numpy.asarray(numpy.floor(numpy.squeeze(hand_points_uvd[:,0])),dtype='int16')]\
                =hand_points_uvd[:,2]


            # plt.figure()
            # plt.imshow(new_hand,'gray')
            # plt.scatter(hand_jnt_gt_uvd[:,0]*bbox_uvd[0,0],hand_jnt_gt_uvd[:,1]*bbox_uvd[0,1])
            # plt.show()

            hand_center_uvd.append([mean_u,mean_v, mean_z])
            uvd_jnt_gt_norm.append(hand_jnt_gt_uvd)

            r0_tmp= interplt.zoom(new_hand, img_size/bbox_uvd[0,0],order=1, mode='nearest',prefilter=True)
            r1_tmp= interplt.zoom(new_hand, img_size/bbox_uvd[0,0]/2,order=1, mode='nearest',prefilter=True)
            r2_tmp=interplt.zoom(new_hand, img_size/bbox_uvd[0,0]/4,order=1, mode='nearest',prefilter=True)
            r0.append(r0_tmp)
            r1.append(r1_tmp)
            r2.append(r2_tmp)

    print 'num of wrong anotation',len(xyz_jnt_gt)-len(new_xyz_jnt_gt)
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
    # dataset = 'test'
    # num_smpl=8252
    # save_path = 'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\NYU_dataset\\NYU_dataset\\test_norm_hand\\norm_hand_uvd_rootmid_zero.h5'
    # label_path = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/NYU_dataset/NYU_dataset/testTable/joint_data.mat'
    dataset = 'train'
    num_smpl=72757
    save_path = 'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\NYU_dataset\\NYU_dataset\\train_norm_hand\\norm_hand_uvd_rootmid.h5'
    label_path = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/NYU_dataset/NYU_dataset/trainTable/joint_data.mat'
    setname='nyu'
    norm_hand(save_path=save_path,setname=setname,
    dataset =dataset,
    fram_prefix = 'depth_1',
    fram_pstfix = '',
    dataset_dir = 'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\NYU_dataset\\NYU_dataset\\%s\\'%dataset,
    num_smpl=num_smpl,
    label_path=label_path)

