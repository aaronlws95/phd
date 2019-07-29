__author__ = 'QiYE'

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt

from src.SphereHandModel.utils import xyz_uvd
from src.SphereHandModel.CostFunction import *

import h5py
import csv


# ImgCenter = [160,120]
# FocalLength = 241.42
# NumOfSampledPoints=512
# depth_ori_path='D:/Project/3DHandPose/Data_3DHandPoseDataset/ICVL_dataset_v2_msrc_format/test/depth/'
# depth_path = 'F:/Proj_Struct_RL_v2/data/icvl/source/DepthImgHandOnly/'
# dist_path = 'F:/Proj_Struct_RL_v2/data/icvl/source/SilhouetteDistImg/'


def getLabelformMsrcFormat(dataset,setname):
    if dataset == 'train':
        path = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/ICVL_dataset_v2_msrc_format/train/log_file_0_v3.csv'
        xyz=  numpy.empty((16008,66),dtype='float32')
    else:
        path = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/ICVL_dataset_v2_msrc_format/test/log_file_0_v3.csv'
        xyz=  numpy.empty((1596,66),dtype='float32')
    print path
    with open(path, 'rb') as f:
        reader = list(csv.reader(f.read().splitlines()))
        for i in xrange(6,len(reader),1):
            xyz[i-6]=reader[i][110:176]
    f.close()

    idx_21 =numpy.array([1,2,3,4,17,5,6,7,18,8,9,10,19,11,12,13,20,14,15,16,21])-1

    xyz.shape = (xyz.shape[0],22,3)
    xyz_21jnt = xyz[:,idx_21,:]*1000
    uvd = xyz_uvd.xyz2uvd(setname=setname, xyz=xyz_21jnt)
    return uvd, xyz_21jnt


def CreateHangImg():
    dataset = 'test'
    setname='icvl'
    dataset_dir = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/ICVL_dataset_v2_msrc_format/%s/depth/'%dataset

    uvd_jnt_gt,xyz_jnt_gt = getLabelformMsrcFormat(dataset,setname)
    for i in xrange(0,xyz_jnt_gt.shape[0],1):
        if i%5000 ==0:
            print i

        file_name = "depth_1_%07d.png" % (i+1)
        depth = Image.open('%s%s' % (dataset_dir, file_name))

        depth = numpy.asarray(depth, dtype='uint32')
        xyz_hand_jnts_gt = numpy.squeeze(xyz_jnt_gt[i]).copy()
        hand_jnt_gt_uvd= xyz_uvd.xyz2uvd(setname=setname,xyz=xyz_hand_jnts_gt)
        #
        # plt.imshow(depth,'gray')
        # plt.scatter(hand_jnt_gt_uvd[:,0],hand_jnt_gt_uvd[:,1])
        # plt.show()
        # # % Collect the points within the AABBOX of the hand
        axis_bounds = numpy.array([numpy.min(hand_jnt_gt_uvd[:, 0]), numpy.max(hand_jnt_gt_uvd[:, 0]),
                                   numpy.min(hand_jnt_gt_uvd[:, 1]), numpy.max(hand_jnt_gt_uvd[:, 1]),
                                   numpy.min(hand_jnt_gt_uvd[:, 2]), numpy.max(hand_jnt_gt_uvd[:, 2])])
        mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
        axis_bounds[mask] -= 20
        axis_bounds[~mask] += 20
        depth_copy= depth.copy()
        handImg = depth_copy[axis_bounds[2]:axis_bounds[3],axis_bounds[0]:axis_bounds[1]]
        loc = numpy.where( (handImg>axis_bounds[4]) & (handImg<axis_bounds[5]))
        handImg[loc]=0
        newDepth = depth-depth_copy

        save_path = 'F:/Proj_Struct_RL_v2/data/icvl/source/DepthImgHandOnly/'
        file_name = "hand_1_%07d" % (i+1)
        img =Image.fromarray(newDepth,mode='I')
        img.save('%s%s.png'%(save_path,file_name))

        # reimg = Image.open('%s%s.png'%(save_path,file_name))
        # reimg = numpy.asarray(reimg, dtype='uint16')
        #
        # plt.figure()
        # plt.imshow(reimg,'gray')
        # # plt.scatter(hand_jnt_gt_uvd[:,0],hand_jnt_gt_uvd[:,1])
        # plt.show()

def SilhouetteDistImg():
    FocalLength = 241.42
    load_path = 'F:/Proj_Struct_RL_v2/data/icvl/source/DepthImgHandOnly/'
    save_path = 'F:/Proj_Struct_RL_v2/data/icvl/source/SilhouetteDistImg/'
    for i in xrange(0,1596,1):


        file_name = "hand_1_%07d" % (i+1)
        silImg = Image.open('%s%s.png'%(load_path,file_name))
        silImg = numpy.asarray(silImg, dtype='uint16')
        tmp = numpy.ones_like(silImg)
        loc = numpy.where(silImg>0)
        tmp[loc]=0
        meanDepth = numpy.mean(silImg[loc])
        """the  distance is measured in pixels and covertered to millimeters using the average input depth """
        silDistImg = numpy.asarray(distance_transform_edt(tmp)*meanDepth/FocalLength,dtype='uint32')
        img =Image.fromarray(silDistImg,mode='I')
        file_name = "dist_1_%07d" % (i+1)
        img.save('%s%s.png'%(save_path,file_name))


        # img2 = Image.open('%s%s.png'%(save_path,file_name))
        # # img = numpy.asarray(img, dtype='uint32')
        #
        # plt.figure()
        # plt.imshow(img2,'gray')
        # # plt.scatter(hand_jnt_gt_uvd[:,0],hand_jnt_gt_uvd[:,1])
        # plt.show()




def getHandOnlyImg(depth,hand_jnt_gt_uvd):
    axis_bounds = numpy.array([numpy.min(hand_jnt_gt_uvd[:, 0]), numpy.max(hand_jnt_gt_uvd[:, 0]),
                               numpy.min(hand_jnt_gt_uvd[:, 1]), numpy.max(hand_jnt_gt_uvd[:, 1]),
                               numpy.min(hand_jnt_gt_uvd[:, 2]), numpy.max(hand_jnt_gt_uvd[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= 20
    axis_bounds[~mask] += 20
    depth_copy= depth.copy()
    handImg = depth_copy[axis_bounds[2]:axis_bounds[3],axis_bounds[0]:axis_bounds[1]]
    loc = numpy.where( (handImg>axis_bounds[4]) & (handImg<axis_bounds[5]))
    handImg[loc]=0
    newDepth = depth-depth_copy
    return newDepth

def getSilhouetteDistImg(handOnlyDepth,FocalLength):
    tmp = numpy.ones_like(handOnlyDepth)
    loc = numpy.where(handOnlyDepth>0)
    tmp[loc]=0
    meanDepth = numpy.mean(handOnlyDepth[loc])
    """the  distance is measured in pixels and covertered to millimeters using the average input depth """
    silDistImg = numpy.asarray(distance_transform_edt(tmp)*meanDepth/FocalLength,dtype='uint32')
    return silDistImg


def evaluateCostFuntion(setname,depth,skeleton,initSphere,ratioForPalm,ratioForFingerSphere,palmBaseTopScaleRatio,FocalLength,
                        ImgCenter,SubPixelNum,ShowCould=False):
    """
        evaluate cost function given a depth input and a skeleton hypothses, the skelton2shpere function
        eg:
    """
    sphere = skeleton2sphere(Skeleton=skeleton, initSphere=initSphere,
                             ratioForPalm=ratioForPalm,ratioForFingerSphere=ratioForFingerSphere,
                             palmBaseTopScaleRatio=palmBaseTopScaleRatio)

    hand_jnt_gt_uvd=xyz2uvd(setname=setname,xyz=skeleton.T)
    handOnlyDepth = getHandOnlyImg(depth=depth,hand_jnt_gt_uvd=hand_jnt_gt_uvd)
    silDistImg = getSilhouetteDistImg(handOnlyDepth,FocalLength)
    cost,term1,term2,term3 = cost_function(setname=setname,DepthImg=handOnlyDepth, Spheres=sphere, Center=ImgCenter,
                         SilhouetteDistImg=silDistImg,SubPixelNum=SubPixelNum)
    # print cost
    if ShowCould==True:
        print 'cost,term1,term2,term3 ', cost,term1,term2,term3
        ShowPointCloudFromDepth(setname=setname,depth=handOnlyDepth,hand_points=skeleton.T,Sphere=sphere.T)
    return cost

def evalaute_icvl_online():

    dataset='test'
    setname='icvl'

    f = h5py.File('F:/Proj_Struct_RL_v2/data/icvl/source/%s_norm_hand_uvd_rootmid_scale_msrcformat.h5'%(dataset), 'r')
    xyz_jnt_gt = f['xyz_jnt_gt_ori'][...]
    f.close()
    hand_model = numpy.load('icvl_hand_sphere_model.npy')

    initSphere=hand_model[0].T

    # for i in xrange(0,10,1):
    for i in numpy.random.randint(0,1000,50):
        file_name = "depth_1_%07d" % (i+1)
        depth = Image.open('%s%s.png'%(depth_ori_path,file_name))
        depth = numpy.asarray(depth, dtype='uint16')
        evaluateCostFuntion(setname=setname,depth=depth,skeleton=xyz_jnt_gt[i].T,initSphere=initSphere.T,ratioForPalm=hand_model[2],
                 ratioForFingerSphere=hand_model[3],palmBaseTopScaleRatio=hand_model[4])
def read_dataset_file(dataset_dir):
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
    xyz_jnt_gt=[]
    file_name = []

    with open('%sTesting_Yang.txt'%(dataset_dir), 'rb') as f:
        for line in f:
            part = line.split('\t')
            file_name.append(part[0].replace('\\', '/'))
            xyz_jnt_gt.append(part[1:64])

    f.close()

    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float32')

    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]

    return file_name,xyz_jnt_gt
def evalaute_mega_online():

    hand_model = numpy.load('F:/HuaweiProj/Proj/data/mega/mega_hand_sphere_model.npy')
    dataset_dir='F:/HuaweiProj/Proj/data/Yang/data/'
    file_name,xyz_joint=read_dataset_file(dataset_dir)
    initSphere=hand_model[0]

    for i in xrange(0,len(file_name),1):
        initSphere[1:4,:]+=xyz_joint[i,9,:].reshape(1,3).T
        depth = Image.open('F:/HuaweiProj/Proj/data/%s' % (file_name[i]))
        depth = numpy.asarray(depth, dtype='uint16')
        tmp = evaluateCostFuntion(setname='mega',depth=depth,skeleton=xyz_joint[i].T,initSphere=initSphere,ratioForPalm=hand_model[2],
                 ratioForFingerSphere=hand_model[3],palmBaseTopScaleRatio=hand_model[4])


def evaluateCostFuntionWsilImgOffline(imgIdx,skeleton,initSphere,ratioForPalm,ratioForFingerSphere,palmBaseTopScaleRatio):
    ImgCenter = [160,120]
    FocalLength = 241.42
    NumOfSampledPoints=512
    depth_ori_path='D:/Project/3DHandPose/Data_3DHandPoseDataset/ICVL_dataset_v2_msrc_format/test/depth/'
    depth_path = 'F:/Proj_Struct_RL_v2/data/icvl/source/DepthImgHandOnly/'
    dist_path = 'F:/Proj_Struct_RL_v2/data/icvl/source/SilhouetteDistImg/'

    sphere = skeleton2sphere(Skeleton=skeleton, initSphere=initSphere,
                             ratioForPalm=ratioForPalm,ratioForFingerSphere=ratioForFingerSphere,
                             palmBaseTopScaleRatio=palmBaseTopScaleRatio)

    file_name = "hand_1_%07d" % (imgIdx+1)
    depthImg = Image.open('%s%s.png'%(depth_path,file_name))
    depthImg = numpy.asarray(depthImg, dtype='uint16')

    file_name = "dist_1_%07d" % (imgIdx+1)
    silImg = Image.open('%s%s.png'%(dist_path,file_name))
    silImg = numpy.asarray(silImg, dtype='uint16')

    cost = cost_function(depthImg, sphere, ImgCenter, silImg)
    print 'cost', cost
    ShowPointCloudFromDepth(depth=depthImg,hand_points=skeleton.T,Sphere=sphere.T)



def evaluate_icvl_offline():
    """
        before call this function, call
        CreateHangImg()
        SilhouetteDistImg()
        to create depth image with only hand and distrance transfromation of the hand mask
        eg:
        if __name__=='__main__':
            CreateHangImg()
            SilhouetteDistImg()
            evaluate_icvl()
    """
    dataset='test'
    f = h5py.File('F:/Proj_Struct_RL_v2/data/icvl/source/%s_norm_hand_uvd_rootmid_scale_msrcformat.h5'%(dataset), 'r')
    xyz_jnt_gt = f['xyz_jnt_gt_ori'][...]
    f.close()
    hand_model = numpy.load('icvl_hand_sphere_model.npy')

    # initSphere=hand_model[0].T
    # initSphere[31]=initSphere[30]

    # for i in xrange(0,10,1):
    for i in numpy.random.randint(0,1000,50):
        evaluateCostFuntionWsilImgOffline(imgIdx=i,skeleton=xyz_jnt_gt[i].T,initSphere=hand_model[0],ratioForPalm=hand_model[2],
                 ratioForFingerSphere=hand_model[3],palmBaseTopScaleRatio=hand_model[4])


if __name__=='__main__':
    evalaute_mega_online()