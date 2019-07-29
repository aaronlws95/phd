__author__ = 'QiYE'
__author__ = 'QiYE'

# Copy right: Shanxin Yuan
# Date: 16 Jan 2017

import numpy
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage.interpolation  as interplt
from matplotlib import pylab
import h5py
from src.utils import convert
import csv
from mpl_toolkits.mplot3d import Axes3D
import scipy.io



def norm_hand(save_path, h5name, matname, dataset_dir, Metadatadir):
    print save_path
    u0= 320
    v0= 240
    img_size = 96.0
    our_index_temp = [1, 2, 5, 8, 11, 14,
                      3, 6, 9, 12, 15,
                      4, 7, 10, 13, 16,
                      17, 18, 19, 20, 21]
    our_index = [x-1 for x in our_index_temp]



    resx = 640
    resy = 480
    fx = resx*0.892592504000000
    fy = resy*1.190123339000000

    ref_z = 1000.0
    # bbsize = 260.0
    bbsize = 200.0
    # addedbound = 20
    addedbound = 20

    center_joint_idx = 3

    xyz_jnt_gt=[]
    file_name = []

    Metadata = sio.loadmat(Metadatadir)
    Camera = Metadata['CameraData']
    Meta = Metadata['MetaData']
    ImgNames = Meta['ImgName']
    JointLocs = Meta['JointLocs']
    TotalNum = JointLocs.shape[1]

    # extract the ground truth joint locations and depth image names
    for f in range(0, TotalNum):
        # print f
        # load depth image
        # depth = Image.open('%s\depth_1_%07d.png' % (dataset_dir, f+1))
        # DepthImg = numpy.asarray(depth, dtype='uint16')

        # load joint locations
        jnts = JointLocs[:,f]
        xyz_jnt_gt.append(jnts[0])

        ImgName = ImgNames[:,f];
        file_name.append(ImgName[0])
        # load image names
        # file_name.append('%s\depth_1_%07d.png' % (dataset_dir, f+1))

    # convert gt to array format
    xyz_jnt_gt = numpy.array(xyz_jnt_gt, dtype='float32')
    print xyz_jnt_gt.shape

    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt =convert.xyz2uvd_Shanxin(xyz_jnt_gt, fx, fy, u0, v0)
    new_xyz_jnt_gt =convert.uvd2xyz_Shanxin(uvd_jnt_gt, fx, fy, u0, v0)
    err =numpy.sqrt(numpy.sum((new_xyz_jnt_gt-xyz_jnt_gt)**2,axis=-1))
    print 'mean err ', numpy.mean(numpy.mean(err))
    bb = numpy.array([(bbsize,bbsize,ref_z)])

    bbox_uvd = convert.xyz2uvd_Shanxin(bb, fx, fy, u0, v0)
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

    # iteration for all images
    for i in xrange(0,xyz_jnt_gt.shape[0],1):
        # print i
        if i%50 ==0:
            print i

        current_file_name = file_name[i]
        depth = Image.open(str(current_file_name[0]))
        # depth = Image.open('%s' % (file_name[i]))
        depth = numpy.asarray(depth, dtype='uint16')
        # print depth.shape

        uvd = convert.convert_depth_to_uvd_Shanxin(depth).astype(dtype='float32')
        xyz = convert.uvd2xyz_Shanxin(uvd, fx, fy, u0, v0)
        points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)
        xyz_hand_jnts_gt = numpy.squeeze(xyz_jnt_gt[i]).copy()

        # % Collect the points within the AABBOX of the hand
        axis_bounds = numpy.array([numpy.min(xyz_hand_jnts_gt[:, 0]), numpy.max(xyz_hand_jnts_gt[:, 0]),
                                   numpy.min(xyz_hand_jnts_gt[:, 1]), numpy.max(xyz_hand_jnts_gt[:, 1]),
                                   numpy.min(xyz_hand_jnts_gt[:, 2]), numpy.max(xyz_hand_jnts_gt[:, 2])])
        # print 'hand bbox in xyz'
        # print numpy.max(xyz_hand_jnts_gt[:, 2]), numpy.min(xyz_hand_jnts_gt[:, 2]),numpy.max(xyz_hand_jnts_gt[:, 2])- numpy.min(xyz_hand_jnts_gt[:, 2])
        # print numpy.max(xyz_hand_jnts_gt[:, 1]), numpy.min(xyz_hand_jnts_gt[:, 1]),numpy.max(xyz_hand_jnts_gt[:, 1])- numpy.min(xyz_hand_jnts_gt[:, 1])
        # print numpy.max(xyz_hand_jnts_gt[:, 0]), numpy.min(xyz_hand_jnts_gt[:, 0]),numpy.max(xyz_hand_jnts_gt[:, 0])-numpy.min(xyz_hand_jnts_gt[:, 0])

        mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
        axis_bounds[mask] -= addedbound
        axis_bounds[~mask] += addedbound
        mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
        mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
        mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
        inputs = mask1 & mask2 & mask3

        points0 = points[inputs]
        new_xyz_jnt_gt.append(xyz_jnt_gt[i])
        new_uvd_jnt_gt.append(uvd_jnt_gt[i])
        new_file_name.append(file_name[i])


        # mean_z = numpy.mean(points0[:,2])
        # mean_z is th avarage depth of the hand point could before moving the hand to the reference depth
        mean_z = xyz_hand_jnts_gt[center_joint_idx,2]
        # now the mean depth value is ref_z
        points0[:,2] += ref_z-mean_z
        xyz_hand_jnts_gt[:,2] += ref_z-mean_z
        hand_points_uvd = convert.xyz2uvd_Shanxin(points0, fx, fy, u0, v0)
        hand_jnt_gt_uvd= convert.xyz2uvd_Shanxin(xyz_hand_jnts_gt, fx, fy, u0, v0)
        # mean_u =numpy.mean(hand_points_uvd[:,0])
        # #mean_u is the avarage u axis width of the hand image after moving the hand to the reference depth
        # mean_v = numpy.mean(hand_points_uvd[:,1])
        # #mean_v is th avarage v axis height of the hand image after moving the hand to the reference depth
        mean_u = hand_jnt_gt_uvd[center_joint_idx,0]
        mean_v = hand_jnt_gt_uvd[center_joint_idx,1]

        # centering u for hand point cloud
        hand_points_uvd[:,0] = hand_points_uvd[:,0] - mean_u+ bbox_uvd[0,0]/2
        hand_points_uvd[ numpy.where(hand_points_uvd[:,0]>=bbox_uvd[0,0]),0]=bbox_uvd[0,0]-1
        hand_points_uvd[ numpy.where(hand_points_uvd[:,0]<0),0]=0
        # centering u for hand joints
        hand_jnt_gt_uvd[:,0] = ( hand_jnt_gt_uvd[:,0] - mean_u + bbox_uvd[0,0]/2 ) / bbox_uvd[0,0]
        hand_jnt_gt_uvd[ numpy.where(hand_jnt_gt_uvd[:,0]>1),0]=1
        hand_jnt_gt_uvd[ numpy.where(hand_jnt_gt_uvd[:,0]<0),0]=0
        # centering v for hand point cloud
        hand_points_uvd[:,1] = hand_points_uvd[:,1] - mean_v+bbox_uvd[0,1]/2
        hand_points_uvd[ numpy.where(hand_points_uvd[:,1]>=bbox_uvd[0,1]),1]=bbox_uvd[0,1]-1
        hand_points_uvd[ numpy.where(hand_points_uvd[:,1]<0),1]=0
        # centering v for hand joints
        hand_jnt_gt_uvd[:,1] =( hand_jnt_gt_uvd[:,1] - mean_v+bbox_uvd[0,1]/2 ) / bbox_uvd[0,1]
        hand_jnt_gt_uvd[ numpy.where(hand_jnt_gt_uvd[:,1]>1),1]=1
        hand_jnt_gt_uvd[ numpy.where(hand_jnt_gt_uvd[:,1]<0),1]=0
        # normalizing d
        hand_points_uvd[:,2] = (hand_points_uvd[:,2] - ref_z + bbsize/2)/bbsize
        hand_jnt_gt_uvd[:,2] = (hand_jnt_gt_uvd[:,2] - ref_z + bbsize/2)/bbsize

        new_hand = numpy.ones((bbox_uvd[0,1],bbox_uvd[0,0]),dtype='float32')
        # resize the hand point cloud
        new_hand[numpy.asarray(numpy.floor(numpy.squeeze(hand_points_uvd[:,1])),dtype='int16'),
                 numpy.asarray(numpy.floor(numpy.squeeze(hand_points_uvd[:,0])),dtype='int16')]\
            =hand_points_uvd[:,2]

        # show the transformation result
        # plt.figure()
        # plt.imshow(new_hand,'gray')
        # plt.scatter(hand_jnt_gt_uvd[:,0],hand_jnt_gt_uvd[:,1])
        # plt.show()

        hand_center_uvd.append([mean_u, mean_v, mean_z])
        # normalize hand joints uv to image_size
        # hand_jnt_gt_uvd[:,0] = hand_jnt_gt_uvd[:,0]*img_size/bbox_uvd[0,0]
        # hand_jnt_gt_uvd[:,1] = hand_jnt_gt_uvd[:,1]*img_size/bbox_uvd[0,1]
        uvd_jnt_gt_norm.append(hand_jnt_gt_uvd)

        r0_tmp= interplt.zoom(new_hand, img_size/bbox_uvd[0,0],order=1, mode='nearest',prefilter=True)
        r1_tmp= interplt.zoom(new_hand, img_size/bbox_uvd[0,0]/2,order=1, mode='nearest',prefilter=True)
        r2_tmp= interplt.zoom(new_hand, img_size/bbox_uvd[0,0]/4,order=1, mode='nearest',prefilter=True)
        r0.append(r0_tmp)
        r1.append(r1_tmp)
        r2.append(r2_tmp)

        # plt.figure()
        # plt.imshow(r0[i],'gray')
        # print 'mean',numpy.mean(r0[i])
        # plt.scatter(hand_jnt_gt_uvd[:,0],hand_jnt_gt_uvd[:,1])
        # plt.show()

    f = h5py.File('%s\%s' % (save_path, h5name), 'w')
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

    sio.savemat('%s\%s' % (save_path, matname), {'HandImg':r0, 'JointLocs':uvd_jnt_gt_norm})


if __name__ == '__main__':

    # training data
    setname='nyu'
    TrainOrTest = 'Train'
    save_path = 'D:\ICCV_2017\Mega\data\\%s' % setname
    h5name1 = '%s_%s_norm_hand_uvd_rootmid.h5' % (setname, TrainOrTest)
    h5name = '%s_%s_norm_hand_uvd_rootmid.h5' % (setname, TrainOrTest)
    matname = '%s_%s_norm_hand_uvd_rootmid.mat' % (setname, TrainOrTest)
    Metadatadir = 'D:\ICCV_2017\Mega\data\\%s\\%s_%s_MetaData.mat' % (setname, setname, TrainOrTest)

    norm_hand(save_path=save_path,
              h5name = h5name,
              matname = matname,
              dataset_dir = 'E:\Dataset\\%s_hand_dataset_v2_msrc_format\Train\depth' % setname,
              Metadatadir = Metadatadir)

