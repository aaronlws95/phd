__author__ = 'QiYE'

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
import numpy
from .adjust_by_dist_cost import cost_function
from . import LabelUtils
import h5py

from ..utils import xyz_uvd


from pyswarm import pso
#####Hand Model Parameter for msrc test######
pointCloudMargin=50
palmBaseTopScaleRatio=0.6
numInSphere=8
numInSphereThumb1=4
FocalLength = 475.0659
setname='mega'
SubPixelNum=512
ImgCenter=[480,640]


def getLabelformMsrcFormat(dataset_dir):

    xyz_jnt_gt=[]
    file_name = []
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
    with open('%s/Training_Annotation.txt'%(dataset_dir), mode='r',encoding='utf-8',newline='') as f:
        for line in f:
            part = line.split('\t')
            file_name.append(part[0])
            xyz_jnt_gt.append(part[1:64])
    f.close()

    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float64')
    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt =xyz_uvd.xyz2uvd(xyz=xyz_jnt_gt,setname='mega')

    return uvd_jnt_gt,xyz_jnt_gt,file_name


def getHandOnlyImg(depth,hand_jnt_gt_uvd):
    axis_bounds = numpy.array([numpy.min(hand_jnt_gt_uvd[:, 0]), numpy.max(hand_jnt_gt_uvd[:, 0]),
                               numpy.min(hand_jnt_gt_uvd[:, 1]), numpy.max(hand_jnt_gt_uvd[:, 1]),
                               numpy.min(hand_jnt_gt_uvd[:, 2]), numpy.max(hand_jnt_gt_uvd[:, 2])],dtype='int16')
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


def read_dataset_file(dataset_dir):
    f = h5py.File('%s/source/test_uvd_xyz_filename.h5'%dataset_dir, 'r')
    uvd =f['uvd'][...]
    xyz =f['xyz'][...]
    filename =f['filename'][...]
    f.close()
    return uvd,xyz,filename


def evalaute_mega_online():


    img_dir = 'F:/BigHand_Challenge/Training'
    save_dir = 'F:/HuaweiProj/data/mega'
    uvd,xyz,file_name=getLabelformMsrcFormat(img_dir)

    Hand_Sphere_Raidus=numpy.array([ 40,
      36,  28,  22,   20,
      34,   26,  22,   20,
      34,   26,  22,   20,
      34,   26,  22,   20,
      34,  26,  22,   20])/2.0

    for i in range(uvd.shape[0]):
        print(i,file_name[i])
        roiDepth =Image.open("%s/images/%s"%(img_dir,file_name[i]))
        depth = numpy.asarray(roiDepth, dtype='uint32')
        cur_xyz=xyz[i]
        cur_uvd=uvd[i]


        sphere = LabelUtils.skeleton2sphere_21jnt(Skeleton=cur_xyz, numInSphere=numInSphere,
                                 palmBaseTopScaleRatio=palmBaseTopScaleRatio,Hand_Sphere_Raidus=Hand_Sphere_Raidus)

        handOnlyDepth = getHandOnlyImg(depth=depth,hand_jnt_gt_uvd=cur_uvd)
        silDistImg = getSilhouetteDistImg(handOnlyDepth,FocalLength)
        cost= cost_function(setname=setname,DepthImg=handOnlyDepth, inSpheres=sphere, Center=ImgCenter,
                             SilhouetteDistImg=silDistImg,SubPixelNum=SubPixelNum)
        # print cost
        print('cost',cost)
        # LabelUtils.ShowPointCloudFromDepth(setname=setname,depth=handOnlyDepth,hand_points=cur_xyz,Sphere=sphere)


        jnt_idx=range(0,21,1)
        selectShpere=sphere[jnt_idx]
        uvd_point_label, partMap,ColorMap = LabelUtils.getPart_mega_21jnt_tmptest(setname=setname,DepthImg=depth,
                                     Sphere=selectShpere.reshape(len(jnt_idx)*numInSphere,5),numInSphere=numInSphere,
                                     numJnt=len(jnt_idx),pointCloudMargin=pointCloudMargin)
        title='%d,%s'%(i,file_name[i])
        # ShowDepthSphere(title=title,setname=setname,depth=depth,Sphere=sphere)
        LabelUtils.ShowDepthSphereMaskHist2(title=title,setname=setname,depth=depth,numPoint=None,jnt_uvd=cur_uvd,hand_points=cur_xyz,Sphere=sphere,partMap=partMap)
        #


# def optimize_pso():
#
#     xopt, fopt = pso(banana, lb, ub, f_ieqcons=con)

if __name__=='__main__':
    evalaute_mega_online()