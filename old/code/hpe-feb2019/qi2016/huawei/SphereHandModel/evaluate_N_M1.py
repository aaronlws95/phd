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

base_dir='F:/pami2017/Proj_CNN_Hier_v2_sx_icvl/'
# base_dir='/home/qi/Proj_CNN_Hier_v2_sx_icvl/'
# xyz_pred_file_name='%sdata/done_v2/icvl_M1_N_Rang10/pso_N29_M1_range10'%(base_dir)
# xyz_pred_file_name='%sdata/done_v2/icvl_M30_N_Rang5/pso_N19_M30_range5'%(base_dir)
# xyz_pred_file_name='%sdata/done_v1/icvl_M500_N1000_Rang10/pso_N99_M500_range10_seg0'%(base_dir)
xyz_pred_file_name='%sdata/done_v1/icvl_M30_N_Rang5/pso_N19_M30_range5'%(base_dir)
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
        # #
        # if ShowCould==True:
        #     print 'cost,term1,term2,term3 ', cost,term1,term2,term3
        #     ShowPointCloudFromDepth2(setname=setname,depth=handOnlyDepth,hand_points=skeleton[i],Sphere=sphere)
    minloc = numpy.argmin(cost_list)

    if ShowCould==True:
        # print('cost',cost_list)
        err=numpy.mean(numpy.sqrt(numpy.sum((skeleton-xyz_gt.reshape(1,21,3))**2,axis=-1)),axis=-1)
        # print('err', err)
        err_cnn =numpy.mean(numpy.mean(numpy.sqrt(numpy.sum((xyz_cnn-xyz_gt)**2,axis=-1)),axis=-1))
        # print('err_golden, err_cnn, err_gt',err[minloc],err_cnn,numpy.min(err))
        # ShowPointCloudFromDepth2(setname=setname,depth=handOnlyDepth,hand_points=skeleton[minloc],Sphere=sphere_list[minloc])
    return cost_list

def evalaute_icvl_online():
    numInSphere=4
    Hand_Sphere_Raidus=numpy.array([ 38,
      36,  30,  24,   20,
      30,   24,    20,   18,
      30,   24,   20,   18,
      30,   24,   20,   18,
      30,  24,  20,    18])/2+4
    setname='icvl'
    palmBaseTopScaleRatio=0.4
    ImgCenter = [160,120]
    FocalLength = 241.42
    NumOfSampledPoints=512*16
    print(NumOfSampledPoints)

    xyz_true =scipy.io.loadmat('%sdata/icvl/source/test_icvl_xyz_21joints.mat'%base_dir)['xyz']*1000
    xyz_cnn = numpy.empty_like(xyz_true)
    for i in range(21):
        xyz_cnn[:,i,:]=scipy.io.loadmat('%sdata/icvl/ICVL_hier_derot_recur_v2/jnt%d_xyz.mat'%(base_dir,i))['jnt']*1000
    mean_err= numpy.mean((numpy.sqrt(numpy.sum((xyz_cnn- xyz_true)**2,axis=-1))))
    print('hiercnn',mean_err)

    img_dir ='%sdata/ICVL_dataset_v2_msrc_format/Testing/depth/'%base_dir


    xyz_pred= scipy.io.loadmat('%s.mat'%xyz_pred_file_name)['jnt']*1000

    cost_matrix=numpy.empty((xyz_pred.shape[0],xyz_pred.shape[1]))
    # for i in numpy.random.randint(0,1000,50):
    for i in range(xyz_pred.shape[0]):
        file_name = "depth_1_%07d" % (i+1)
        print(i)
        depth = Image.open('%s%s.png'%(img_dir,file_name))
        depth = numpy.asarray(depth, dtype='uint16')

        cost_list=evaluateCostFuntion(setname=setname,depth=depth,skeleton=xyz_pred[i],xyz_gt=xyz_true[i],xyz_cnn=xyz_cnn[i],
                            numInSphere=numInSphere,palmBaseTopScaleRatio=palmBaseTopScaleRatio,
                            FocalLength=FocalLength,
                        ImgCenter=ImgCenter,SubPixelNum=NumOfSampledPoints,Hand_Sphere_Raidus=Hand_Sphere_Raidus)
        cost_matrix[i]=cost_list
    numpy.save('%s_cost.npy'%(xyz_pred_file_name),cost_matrix)
    # cost_matrix=numpy.load('%sdata/done/icvl_M30_N_Rang5/cost_pso_N19_M30_range10.npy')

    err=[]
    for i in range(1,20,1):
        minloc=numpy.argmin(cost_matrix[:,:i],axis=1)
        pso_result=xyz_pred[:,:i][(range(cost_matrix.shape[0]),minloc)]
        # print(cost_matrix[:,:i].shape,pso_result.shape)
        mean_err= numpy.mean((numpy.sqrt(numpy.sum((pso_result- xyz_true)**2,axis=-1))))
        err.append(mean_err)
    print(err)
    numpy.save('%s_err.npy'%(xyz_pred_file_name),err)
    # plt.bar(range(len(err)),err)
    # plt.savefig('%s_err.png'%(xyz_pred_file_name))
def draw_M_N():

    err_ori = numpy.load('%s_err.npy'%(xyz_pred_file_name))
    print(err_ori)
    err=err_ori
    plt.bar(range(err.shape[0]),err)
    plt.savefig('%s_err.png'%(xyz_pred_file_name))
    # plt.show()
def analy_err():
    xyz_true =scipy.io.loadmat('%sdata/icvl/source/test_icvl_xyz_21joints.mat'%base_dir)['xyz']*1000
    xyz_cnn = numpy.empty_like(xyz_true)
    for i in range(21):
        xyz_cnn[:,i,:]=scipy.io.loadmat('F:/pami2017/Prj_CNN_Hier_v2/data/icvl/ICVL_hier_derot_recur_v2/jnt%d_xyz.mat'%(i))['jnt']*1000
    mean_err= numpy.mean((numpy.sqrt(numpy.sum((xyz_cnn- xyz_true)**2,axis=-1))))
    print('hiercnn',mean_err)
    cost_matrix=numpy.load('%s_cost.npy'%(xyz_pred_file_name))
    minloc=numpy.argmin(cost_matrix,axis=1)
    # print('cost',cost_list)
    xyz_pred = scipy.io.loadmat('%s.mat'%(xyz_pred_file_name))['jnt']*1000
    xyz_min=xyz_pred[(range(cost_matrix.shape[0]),minloc)]

    err_pso=numpy.sqrt(numpy.sum((xyz_min-xyz_true)**2,axis=-1))
    err_jnt_pso=err_pso.flatten()
    err_cnn =numpy.sqrt(numpy.sum((xyz_cnn-xyz_true)**2,axis=-1))
    err_jnt_cnn=err_cnn.flatten()

    loc = numpy.where(err_jnt_cnn>40)
    print(loc[0].shape[0]*1.0/err_jnt_cnn.shape[0])
    print(numpy.mean(err_jnt_cnn[loc]))
    print(numpy.mean(err_jnt_pso[loc]))

    err_gt=numpy.mean(numpy.sqrt(numpy.sum((xyz_pred-xyz_true.reshape(1596,1,21,3))**2,axis=-1)),axis=-1)
    print(err_gt.shape)
    err_list_gt=[]
    for i in range(2,xyz_pred.shape[1],1):
        minloc=numpy.argmin(err_gt[:,:i],axis=1)
        print(err_gt[:,:i][minloc].shape)
        tmperrr=numpy.mean(err_gt[:,:i][(range(cost_matrix.shape[0]),minloc)])
        err_list_gt.append(tmperrr)
    plt.bar(range(len(err_list_gt)),err_list_gt)
    plt.savefig('%s_err_chosenbygt.png'%(xyz_pred_file_name))
    print(err_list_gt)
    print(numpy.mean(err_jnt_cnn))
    print(numpy.mean(err_jnt_pso))
if __name__=='__main__':
    # evalaute_icvl_online()
    # draw_M_N()
    analy_err()