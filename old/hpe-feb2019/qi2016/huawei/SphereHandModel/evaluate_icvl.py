__author__ = 'QiYE'

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from src.SphereHandModel.utils import xyz_uvd
from src.SphereHandModel.CostFunction import cost_function
from src.SphereHandModel.ShowSamples import ShowPointCloudFromDepth2
import scipy.io
import numpy
import csv
THUMB=[1,2,3,4]
INDEX=[5,6,7,8]
MIDDLE=[9,10,11,12]
RING=[13,14,15,16]
PINKY=[17,18,19,20]
WRIST=0
PALM=[1,5,9,13,17]




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
        print('err_golden, err_cnn, err_gt',err[minloc],err_cnn,numpy.min(err))
        # ShowPointCloudFromDepth2(setname=setname,depth=handOnlyDepth,hand_points=skeleton[minloc],Sphere=sphere_list[minloc])
    return cost_list

def evalaute_icvl_online():
    numInSphere=4
    Hand_Sphere_Raidus=numpy.array([ 38,
      36,  30,  24,   20,
      30,   24,    20,   18,
      30,   24,   20,   18,
      30,   24,   20,   18,
      30,  24,  20,    18])/2-2
    setname='icvl'
    palmBaseTopScaleRatio=0.4
    ImgCenter = [160,120]
    FocalLength = 241.42
    NumOfSampledPoints=512

    xyz_true =scipy.io.loadmat('F:/pami2017/Proj_CNN_Hier_v2_sx_icvl/data/icvl/source/test_icvl_xyz_21joints.mat')['xyz']*1000
    xyz_pred = scipy.io.loadmat('F:/pami2017/Proj_CNN_Hier_v2_sx_icvl/data/icvl_M1_N_Rang5/pso_N104_M1_range5.mat')['jnt']*1000
    xyz_cnn = numpy.empty_like(xyz_true)
    for i in range(21):
        xyz_cnn[:,i,:]=scipy.io.loadmat('F:/pami2017/Prj_CNN_Hier_v2/data/icvl/ICVL_hier_derot_recur_v2/jnt%d_xyz.mat'%i)['jnt']*1000

    mean_err= numpy.mean((numpy.sqrt(numpy.sum((xyz_cnn- xyz_true)**2,axis=-1))))

    print('hiercnn',mean_err)
    img_dir ='F:/pami2017/Proj_CNN_Hier_v2_sx_icvl/data/ICVL_dataset_v2_msrc_format/Testing/depth/'

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
    numpy.save('F:/pami2017/Proj_CNN_Hier_v2_sx_icvl/data/icvl_M1_N_Rang5/cost_pso_N104_M1_range5_term1_more.npy',cost_matrix)
    # cost_matrix=numpy.load('F:/pami2017/Proj_CNN_Hier_v2_sx_icvl/data/done/icvl_M30_N_Rang5/cost_pso_N19_M30_range5.npy')
    err_list=[]
    for i in range(1,105,1):
        minloc=numpy.argmin(cost_matrix[:,:i],axis=1)
        pso_result=xyz_pred[:,:i][(range(cost_matrix.shape[0]),minloc)]
        # print(cost_matrix[:,:i].shape,pso_result.shape)
        mean_err= numpy.mean((numpy.sqrt(numpy.sum((pso_result- xyz_true)**2,axis=-1))))
        err_list.append(mean_err)
    print(range(1,20,1))
    print(err_list)

if __name__=='__main__':
    evalaute_icvl_online()