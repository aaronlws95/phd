import scipy.io
from ..utils import xyz_uvd
import csv
from PIL import Image
import numpy
import matplotlib.pyplot as plt
from .LabelUtils import skeleton2sphere_21jnt,getPart_mega_21jnt,ShowDepthSphere,ShowDepthSphereMaskHist2,getPart_mega_21jnt_tmptest
import h5py

#####Hand Model Parameter for msrc test######
pointCloudMargin=50
palmBaseTopScaleRatio=0.6
numInSphere=10
numInSphereThumb1=4



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

def get_mean_of_hand_points(depth,setname,joint_xyz):
    uvd = xyz_uvd.convert_depth_to_uvd(depth)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    axis_bounds = numpy.array([numpy.min(joint_xyz[:, 0]), numpy.max(joint_xyz[:, 0]),
                               numpy.min(joint_xyz[:, 1]), numpy.max(joint_xyz[:, 1]),
                               numpy.min(joint_xyz[:, 2]), numpy.max(joint_xyz[:, 2])])
    margin=30
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= margin
    axis_bounds[~mask] += margin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    return numpy.mean(points0,axis=0)


def getPart_mega():

    setname='mega'
    anno_dir='F:/BigHand_Challenge/Training'
    lable_dir='F:/BigHand_Challenge/Training/partlabels'
    img_dir = 'F:/BigHand_Challenge/Training/images'
    uvd, xyz_joint,file_name =getLabelformMsrcFormat(anno_dir)

    Hand_Sphere_Raidus=numpy.array([ 36,
      33,  28,  22,   20,
      34,   20,  20,   20,
      32,   20,  20,   20,
      32,   20,  20,   20,
      32,  20,  20,   20])/2.0

    for i in range(450000,uvd.shape[0],1):
    # for i in range(10495,xyz_joint.shape[0],1):
    # for i in numpy.random.randint(0,uvd.shape[0],50):
        print(i,file_name[i])
        roiDepth =Image.open("%s/%s"%(img_dir,file_name[i]))
        depth = numpy.asarray(roiDepth, dtype='uint32')
        cur_xyz=xyz_joint[i]
        cur_uvd=uvd[i]


        sphere = skeleton2sphere_21jnt(Skeleton=cur_xyz, numInSphere=numInSphere,
                                 palmBaseTopScaleRatio=palmBaseTopScaleRatio,Hand_Sphere_Raidus=Hand_Sphere_Raidus)
        mean_sphere = numpy.mean(sphere[:,1:4],axis=0)
        mean_points = get_mean_of_hand_points(depth=depth,setname=setname,joint_xyz=cur_xyz)



if __name__=='__main__':

    getPart_mega()
        # labelRefPose_mega(dataset=set)



