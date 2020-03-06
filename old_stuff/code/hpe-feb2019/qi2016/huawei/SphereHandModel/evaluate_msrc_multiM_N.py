__author__ = 'QiYE'

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from src.SphereHandModel.utils import xyz_uvd
from src.SphereHandModel.CostFunction import cost_function
from src.SphereHandModel.ShowSamples import ShowPointCloudFromDepth2
import scipy.io
import numpy
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
THUMB=[1,2,3,4]
INDEX=[5,6,7,8]
MIDDLE=[9,10,11,12]
RING=[13,14,15,16]
PINKY=[17,18,19,20]
WRIST=0
PALM=[1,5,9,13,17]

base_dir='F:/pami2017/Proj_CNN_Hier_v2_sx_msrc/'
# base_dir='/home/qi/Proj_CNN_Hier_v2_sx_msrc/'
#

def getHandOnlyImg(depth,hand_jnt_gt_uvd):
    pad_width=100
    axis_bounds = numpy.array([numpy.min(hand_jnt_gt_uvd[:, 0]), numpy.max(hand_jnt_gt_uvd[:, 0]),
                               numpy.min(hand_jnt_gt_uvd[:, 1]), numpy.max(hand_jnt_gt_uvd[:, 1]),
                               numpy.min(hand_jnt_gt_uvd[:, 2]), numpy.max(hand_jnt_gt_uvd[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= 20
    axis_bounds[~mask] += 20
    depth0=numpy.empty((depth.shape[0]+pad_width*2,depth.shape[1]+pad_width*2))
    depth0[pad_width:(pad_width+depth.shape[0]),pad_width:(pad_width+depth.shape[1])]=depth
    depth_copy= depth0.copy()
    handImg = depth_copy[(int(axis_bounds[2])+pad_width):(int(axis_bounds[3])+pad_width),(int(axis_bounds[0])+pad_width):(int(axis_bounds[1])+pad_width)]
    loc = numpy.where( (handImg>axis_bounds[4]) & (handImg<axis_bounds[5]))
    handImg[loc]=0
    newDepth = depth0-depth_copy
    return newDepth[pad_width:(pad_width+depth.shape[0]),pad_width:(pad_width+depth.shape[1])]

def getSilhouetteDistImg(handOnlyDepth,FocalLength):
    tmp = numpy.ones_like(handOnlyDepth)
    loc = numpy.where(handOnlyDepth>0)
    tmp[loc]=0
    meanDepth = numpy.mean(handOnlyDepth[loc])
    """the  distance is measured in pixels and covertered to millimeters using the average input depth """
    silDistImg = numpy.asarray(distance_transform_edt(tmp)*meanDepth/FocalLength,dtype='uint32')
    return silDistImg

def skeleton2sphere(Skeleton,numInSphere,palmBaseTopScaleRatio,Hand_Sphere_Raidus):

    newSphere = numpy.empty((21,numInSphere,5))


    fingerIdxList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    ##thumb joints interpolate
    for i in range(21):
        newSphere[i,:,0]=i
    for j in range(numInSphere):

        newSphere[1,j,1:4]= (Skeleton[1]-Skeleton[WRIST])*(j+1)/numInSphere+Skeleton[WRIST]
        newSphere[1,j,4]=(Hand_Sphere_Raidus[1,]-Hand_Sphere_Raidus[WRIST,])*(j+1)/numInSphere+Hand_Sphere_Raidus[WRIST,]

    ##finger joint interpolate
    for skeIdx in fingerIdxList:
        for i in range(len(THUMB)-1):

            for j in range(numInSphere):
                newSphere[skeIdx[i+1],j,1:4]=(Skeleton[skeIdx[i+1]]-Skeleton[skeIdx[i]])*(j+1)/numInSphere+Skeleton[skeIdx[i]]
                newSphere[skeIdx[i+1],j,4]=(Hand_Sphere_Raidus[skeIdx[i+1],]-Hand_Sphere_Raidus[skeIdx[i],])*(j+1)/numInSphere+Hand_Sphere_Raidus[skeIdx[i],]

    ##tralate palm joint to wrist get the other end of line
    idx=[5,9,13,17]
    centre_idx=5
    scaleMat = numpy.eye(3)*palmBaseTopScaleRatio
    tmpSke = Skeleton-Skeleton[centre_idx].reshape(1,3)
    basePalmJoints = numpy.dot(tmpSke[idx],scaleMat)+tmpSke[0].reshape(1,3)+Skeleton[centre_idx].reshape(1,3)

    for j in range(numInSphere):

        newSphere[0,j,1:4]= (basePalmJoints[-1]-basePalmJoints[0])*(j+1)/numInSphere+basePalmJoints[0]
        newSphere[0,j,4]=Hand_Sphere_Raidus[WRIST]



    ##palm joints except thumb joint interpolate
    for i, idx in enumerate(PALM[1:]):

        for j in range(numInSphere):
            newSphere[idx,j,1:4]= (Skeleton[idx]-basePalmJoints[i])*(j+1)/numInSphere+basePalmJoints[i]
            newSphere[idx,j,4]=(Hand_Sphere_Raidus[idx,]-Hand_Sphere_Raidus[WRIST,])*(j+1)/numInSphere+Hand_Sphere_Raidus[WRIST,]
    return newSphere


def evaluateCostFuntion(setname,depth,skeleton,xyz_gt, xyz_cnn,numInSphere,palmBaseTopScaleRatio,FocalLength,
                        ImgCenter,SubPixelNum,Hand_Sphere_Raidus,ShowCould=True):
    """
        evaluate cost function given a depth input and a skeleton hypothses, the skelton2shpere function
        eg:
    """
    # sphere = skeleton2sphere(Skeleton=skeleton, initSphere=initSphere,
    #                          ratioForPalm=ratioForPalm,ratioForFingerSphere=ratioForFingerSphere,
    #                          palmBaseTopScaleRatio=palmBaseTopScaleRatio)
    hand_jnt_gt_uvd=xyz_uvd.xyz2uvd(setname=setname,xyz=xyz_gt)
    handOnlyDepth = getHandOnlyImg(depth=depth,hand_jnt_gt_uvd=hand_jnt_gt_uvd)
    silDistImg = getSilhouetteDistImg(handOnlyDepth,FocalLength)
    cost_list=[]
    sphere_list=[]
    # N_list=[5,10,15,20]
    for i in range(skeleton.shape[0]):
        sphere = skeleton2sphere(Skeleton=skeleton[i], numInSphere=numInSphere,
                                 palmBaseTopScaleRatio=palmBaseTopScaleRatio,Hand_Sphere_Raidus=Hand_Sphere_Raidus)
        sphere_list.append(sphere)
        cost,term1,term2,term3 = cost_function(setname=setname,DepthImg=handOnlyDepth, inSpheres=sphere, Center=ImgCenter,
                             SilhouetteDistImg=silDistImg,SubPixelNum=SubPixelNum)
        # print cost
        cost_list.append(cost)
        #
        # if ShowCould==True:
        #     print 'cost,distance term1,background term2,conflict term3 ', cost,term1,term2,term3
        #     ShowPointCloudFromDepth2(setname=setname,depth=handOnlyDepth,hand_points=skeleton[i],Sphere=sphere)
    # minloc = numpy.argmin(cost_list)
    #
    # if ShowCould==True:
    #     # print('cost',cost_list)
    #     err=numpy.mean(numpy.sqrt(numpy.sum((skeleton-xyz_gt.reshape(1,21,3))**2,axis=-1)),axis=-1)
    #     # print('err', err)
    #     err_cnn =numpy.mean(numpy.mean(numpy.sqrt(numpy.sum((xyz_cnn-xyz_gt)**2,axis=-1)),axis=-1))
    #     # print('err_golden, err_cnn, err_gt',err[minloc],err_cnn,numpy.min(err))
    #     # ShowPointCloudFromDepth2(setname=setname,depth=handOnlyDepth,hand_points=skeleton[minloc],Sphere=sphere_list[minloc])
    return cost_list

def evalaute_msrc_online():
    numInSphere=4
    Hand_Sphere_Raidus=numpy.array([ 36,
      32,  24,  20,   16,
      30,   20,    16,   12,
      30,   20,   16,   12,
      30,   20,   16,   12,
      30,  20,  16,    12])/2
    setname='msrc'
    palmBaseTopScaleRatio=0.4
    ImgCenter = [256,212]
    FocalLength = 356
    NumOfSampledPoints=512


    xyz_true =scipy.io.loadmat('%sdata/msrc/source/test_msrc_xyz_21joints.mat'%base_dir)['xyz']*1000
    xyz_cnn = numpy.empty_like(xyz_true)
    for i in range(21):
        xyz_cnn[:,i,:]=scipy.io.loadmat('%sdata/msrc_hier_derot_recur_v2/jnt%d_xyz.mat'%(base_dir,i))['jnt']*1000
    mean_err= numpy.mean((numpy.sqrt(numpy.sum((xyz_cnn- xyz_true)**2,axis=-1))))
    print('hiercnn',mean_err)

    img_dir ='%sdata/MSRC_dataset_v2_msrc_format/Testing/depth/'%base_dir
    range_val = 10
    M_list=[45]
    # M_list=[3,15,30,45]
    N_max=100-1
    err_matrix=numpy.empty((len(M_list),N_max))

    for im, M in enumerate(M_list):
        print('M',M)
        xyz_pred_file_name='%sdata/msrc_M%d_N_Rang%d/pso_N%d_M%d_range%d'%(base_dir,M,range_val,N_max,M,range_val)
        xyz_pred = scipy.io.loadmat('%s.mat'%(xyz_pred_file_name))['jnt']*1000
        cost_matrix=numpy.empty((xyz_pred.shape[0],xyz_pred.shape[1]))
        # for i in numpy.random.randint(0,1000,50):
        for i in range(xyz_pred.shape[0]):
            # print()
            file_name = "testds_%06d_depth" % (i)
            print('%sdata/msrc_M%d_N_Rang%d/pso_N%d_M%d_range%d'%(base_dir,M,range_val,N_max,M,range_val), file_name)
            depth = Image.open('%s%s.png'%(img_dir,file_name))
            depth = numpy.asarray(depth, dtype='uint16')

            cost_list=evaluateCostFuntion(setname=setname,depth=depth,skeleton=xyz_pred[i],xyz_gt=xyz_true[i],xyz_cnn=xyz_cnn[i],
                                numInSphere=numInSphere,palmBaseTopScaleRatio=palmBaseTopScaleRatio,
                                FocalLength=FocalLength,
                            ImgCenter=ImgCenter,SubPixelNum=NumOfSampledPoints,Hand_Sphere_Raidus=Hand_Sphere_Raidus)
            cost_matrix[i]=cost_list
        numpy.save('%sdata/MNcost/M%d_N%d_cost_range%d.npy'%(base_dir,M,N_max,range_val),cost_matrix)
        # cost_matrix=numpy.load('%sdata/done/msrc_M30_N_Rang5/cost_pso_N19_M30_range10.npy')

    #     for i in range(1,N_max,1):
    #         minloc=numpy.argmin(cost_matrix[:,:i],axis=1)
    #         pso_result=xyz_pred[:,:i][(range(cost_matrix.shape[0]),minloc)]
    #         # print(cost_matrix[:,:i].shape,pso_result.shape)
    #         mean_err= numpy.mean((numpy.sqrt(numpy.sum((pso_result- xyz_true)**2,axis=-1))))
    #         err_matrix[im,i]=mean_err
    # numpy.save('%sdata/MNcost/M%dto%d_N%d_err.npy'%(base_dir,M_list[0],M_list[-1],N_max),err_matrix)
    
def get_min_err_by_cost():

    M_list=[3,15,30,45]
    N_max=100-1
    range_val = 20
    xyz_true =scipy.io.loadmat('%sdata/msrc/source/test_msrc_xyz_21joints.mat'%base_dir)['xyz']*1000
    err_matrix=numpy.empty((4,N_max))
    for im, M in enumerate(M_list):
        xyz_pred_file_name='%sdata/msrc_M%d_N_Rang%d/pso_N%d_M%d_range%d'%(base_dir,M,range_val,N_max,M,range_val)
        cost_matrix= numpy.load('%sdata/MNcost/M%d_N%d_cost_range%d.npy'%(base_dir,M,N_max,range_val))
        # cost_matrix=numpy.load('%sdata/done/msrc_M30_N_Rang5/cost_pso_N19_M30_range10.npy')
        xyz_pred = scipy.io.loadmat('%s.mat'%(xyz_pred_file_name))['jnt']*1000
        for i in range(1,N_max+1,1):
            minloc=numpy.argmin(cost_matrix[:,:i],axis=1)
            pso_result=xyz_pred[:,:i][(range(cost_matrix.shape[0]),minloc)]
            # print(cost_matrix[:,:i].shape,pso_result.shape)
            mean_err= numpy.mean((numpy.sqrt(numpy.sum((pso_result- xyz_true)**2,axis=-1))))
            err_matrix[im,i-1]=mean_err
    numpy.save('%sdata/MNcost_nomean/M%dto%d_N%d_err_range%d.npy'%(base_dir,M_list[0],M_list[-1],N_max,range_val),err_matrix)
    print('%sdata/MNcost_nomean/M%dto%d_N%d_err_range%d.npy'%(base_dir,M_list[0],M_list[-1],N_max,range_val))
def get_min_err_by_gt():

    M_list=[3,15,30,45]
    N_max=54
    range_val = 5
    xyz_true =scipy.io.loadmat('%sdata/msrc/source/test_msrc_xyz_21joints.mat'%base_dir)['xyz']*1000
    err_matrix=numpy.empty((len(M_list),N_max))
    for im, M in enumerate(M_list):
        xyz_pred_file_name='%sdata/MN_mean/msrc_M%d_Rang%d/pso_N%d_M%d_range%d'%(base_dir,M,range_val,N_max,M,range_val)
        # cost_matrix= numpy.load('%sdata/MNcost/M%d_N%d_cost_range%d.npy'%(base_dir,M,N_max,range_val))
        # cost_matrix=numpy.load('%sdata/done/msrc_M30_N_Rang5/cost_pso_N19_M30_range10.npy')
        xyz_pred = scipy.io.loadmat('%s.mat'%(xyz_pred_file_name))['jnt']*1000
        tmp = numpy.sum((xyz_pred-xyz_true.reshape(xyz_true.shape[0],1,21,3))**2,axis=-1)
        cost_matrix= numpy.mean(numpy.sqrt(tmp),axis=-1)
        for i in range(1,N_max+1,1):
            minloc=numpy.argmin(cost_matrix[:,:i],axis=1)
            pso_result=xyz_pred[:,:i][(range(cost_matrix.shape[0]),minloc)]
            # print(cost_matrix[:,:i].shape,pso_result.shape)
            mean_err= numpy.mean((numpy.sqrt(numpy.sum((pso_result- xyz_true)**2,axis=-1))))
            err_matrix[im,i-1]=mean_err
    numpy.save('%sdata/MN_mean/MNcost_mean/M%dto%d_N%d_gt_err_range%d.npy'%(base_dir,M_list[0],M_list[-1],N_max,range_val),err_matrix)
    print('%sdata/MN_mean/MNcost_mean/M%dto%d_N%d_gt_err_range%d.npy'%(base_dir,M_list[0],M_list[-1],N_max,range_val))


def dawnbar(err_typ):
    M_list=[3,15,30,45]
    N_max=54
    range_val = 5
    idx =[0,9,19,29,39,49]
    err_matrix_ori=numpy.load('%sdata/MN_mean/MNcost_mean/M%dto%d_N%d_%s_range%d.npy'%(base_dir,M_list[0],M_list[-1],N_max,err_typ,range_val))
    print(idx)
    err_matrix=err_matrix_ori[:,idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xidx=[3,15,30,45]
    print(err_matrix)
    xpos, ypos = numpy.meshgrid(xidx,idx)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = numpy.zeros_like(xpos)
# Construct arrays with the dimensions for the 16 bars.
    dx = 10 * np.ones_like(zpos)
    dy = 1* np.ones_like(zpos)
    dz =err_matrix.flatten()
    ax.set_xlabel('M')
    ax.set_ylabel('N')
    ax.set_xticks(xidx)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    # plt.savefig('%sdata/MNcost/M%dto%d_N%d_err.png'%(base_dir,M_list[0],M_list[-1],N_max))
    # plt.savefig('%sdata/done_v3/msrc_M3_45_N_Rang5/err.png'%base_dir)
    plt.show()

import numpy as np
def test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.random.rand(2, 100) * 4
    hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

    # Construct arrays for the anchor positions of the 16 bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
    # with indexing='ij'.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    plt.show()


def find_hand_model():
    numInSphere=4
    Hand_Sphere_Raidus=numpy.array([ 36,
      32,  24,  20,   16,
      30,   20,    16,   12,
      30,   20,   16,   12,
      30,   20,   16,   12,
      30,  20,  16,    12])/2
    setname='msrc'
    palmBaseTopScaleRatio=0.4
    ImgCenter = [256,212]
    FocalLength = 356
    NumOfSampledPoints=512


    xyz_true =scipy.io.loadmat('%sdata/msrc/source/test_msrc_xyz_21joints.mat'%base_dir)['xyz']*1000


    img_dir ='%sdata/MSRC_dataset_v2_msrc_format/Testing/depth/'%base_dir

    for i in range(xyz_true.shape[0]):
        file_name = "testds_%06d_depth" % (i)
        # print(i)
        depth = Image.open('%s%s.png'%(img_dir,file_name))
        depth = numpy.asarray(depth, dtype='uint16')

        cost_list=evaluateCostFuntion(setname=setname,depth=depth,skeleton=xyz_true[i].reshape(1,21,3),xyz_gt=xyz_true[i],xyz_cnn=xyz_true[i],
                            numInSphere=numInSphere,palmBaseTopScaleRatio=palmBaseTopScaleRatio,
                            FocalLength=FocalLength,
                        ImgCenter=ImgCenter,SubPixelNum=NumOfSampledPoints,Hand_Sphere_Raidus=Hand_Sphere_Raidus)

def get_min_err_by_gt_1M_mean(M):

    # M_list=[3,15,30,45]
    N_max=99
    range_val = 15
    xyz_true =scipy.io.loadmat('%sdata/msrc/source/test_msrc_xyz_21joints.mat'%base_dir)['xyz']*1000
    err_matrix=numpy.empty((1,N_max))

    xyz_pred_file_name='%sdata/MN_mean/msrc_M%d_N_Rang%d/pso_N%d_M%d_range%d'%(base_dir,M,range_val,N_max,M,range_val)
    xyz_pred = scipy.io.loadmat('%s.mat'%(xyz_pred_file_name))['jnt']*1000
    tmp = numpy.sum((xyz_pred-xyz_true.reshape(xyz_true.shape[0],1,21,3))**2,axis=-1)
    cost_matrix= numpy.mean(numpy.sqrt(tmp),axis=-1)
    for i in range(1,N_max+1,1):
        minloc=numpy.argmin(cost_matrix[:,:i],axis=1)
        pso_result=xyz_pred[:,:i][(range(cost_matrix.shape[0]),minloc)]
        # print(cost_matrix[:,:i].shape,pso_result.shape)
        mean_err= numpy.mean((numpy.sqrt(numpy.sum((pso_result- xyz_true)**2,axis=-1))))
        err_matrix[0,i-1]=mean_err
    numpy.save('%sdata/MN_mean/MNcost_mean/M%d_N%d_gt_err_range%d.npy'%(base_dir,M,N_max,range_val),err_matrix)
    print(err_matrix[0,0:7])
if __name__=='__main__':
    # test()
    # find_hand_model()
    # # evalaute_msrc_online()
    # get_min_err_by_cost()
    # get_min_err_by_gt()
    # dawnbar(err_typ='gt_err')
    get_min_err_by_gt_1M_mean(M=50)