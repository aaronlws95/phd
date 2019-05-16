
__author__ = 'QiYE'

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt

from src.SphereHandModel.utils import xyz_uvd
from src.rl_icvl.SkeletonSphereMotion import *


WRIST=0
PALM=[1,5,9,13,17]
MCP=[2,6,10,14,18]
DIP=[3,7,11,15,19]
TIP=[4,8,12,16,20]

THUMB=[1,2,3,4]
INDEX=[5,6,7,8]
MIDDLE=[9,10,11,12]
RING=[13,14,15,16]
PINKY=[17,18,19,20]

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

if __name__ == '__main__':

    SilhouetteDistImg()

    # CreateHangImg()


