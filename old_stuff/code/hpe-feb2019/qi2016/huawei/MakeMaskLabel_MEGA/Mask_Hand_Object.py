import scipy.io
from ..utils import xyz_uvd
import csv
from PIL import Image
import numpy
import matplotlib.pyplot as plt
from . import LabelUtils
import h5py

#####Hand Model Parameter for msrc test######
pointCloudMargin=50
palmBaseTopScaleRatio=0.6
numInSphere=8
numInSphereThumb1=4



def getLabelformMsrcFormat(dataset_dir):

    xyz_jnt_gt=[]
    file_name = []
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
    with open('%s/Annotation.txt'%(dataset_dir), mode='r',encoding='utf-8',newline='') as f:
        for line in f:
            part = line.split('\t')
            file_name.append(part[0])
            xyz_jnt_gt.append(part[1:64])
    f.close()

    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float64')
    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt =xyz_uvd.xyz2uvd(xyz=xyz_jnt_gt,setname='mega')
    print(uvd_jnt_gt.shape)

    return uvd_jnt_gt,xyz_jnt_gt,file_name


def getPart_mega():

    setname='mega'
    anno_dir='F:/BigHand_Challenge/hand_object'
    # lable_dir='F:/BigHand_Challenge/Training/partlabels'
    img_dir = 'F:/BigHand_Challenge/hand_object/images'
    uvd, xyz_joint,file_name =getLabelformMsrcFormat(anno_dir)

    Hand_Sphere_Raidus=numpy.array([ 40,
      36,  34,  30,   30,
      34,   30,  26,   26,
      34,   30,  26,   26,
      34,   30,  26,   26,
      34,  30,  26,   26])/2.0



    for i in range(15,uvd.shape[0],1):
    # for i in range(10495,xyz_joint.shape[0],1):
    # for i in numpy.random.randint(0,uvd.shape[0],50):
        print(i,file_name[i])
        roiDepth =Image.open("%s/%s"%(img_dir,file_name[i]))
        depth = numpy.asarray(roiDepth, dtype='uint32')
        cur_xyz=xyz_joint[i]
        cur_uvd=uvd[i]


        sphere = LabelUtils.skeleton2sphere_21jnt(Skeleton=cur_xyz, numInSphere=numInSphere,
                                 palmBaseTopScaleRatio=palmBaseTopScaleRatio,Hand_Sphere_Raidus=Hand_Sphere_Raidus)
        # ShowDepthSphere(title='%d,%s'%(i,file_name[i]),setname=setname,depth=depth,Sphere=sphere)
        # jnt_idx=range(0,21,1)
        # selectShpere=sphere[jnt_idx]
        # partMap,ColorMap = getPart_mega_ori(setname=setname,DepthImg=depth,
        #                              Sphere=selectShpere.reshape(len(jnt_idx)*numInSphere,5),numInSphere=numInSphere,
        #                              numJnt=len(jnt_idx),pointCloudMargin=pointCloudMargin)
        # plt.imshow(ColorMap,'jet')
        # plt.show()
        jnt_idx=range(0,21,1)
        selectShpere=sphere[jnt_idx]
        uvd_point_label, partMap,ColorMap = LabelUtils.getPart_mega_21jnt_handobject(setname=setname,DepthImg=depth,
                                     Sphere=selectShpere.reshape(len(jnt_idx)*numInSphere,5),numInSphere=numInSphere,
                                     numJnt=len(jnt_idx),pointCloudMargin=pointCloudMargin)
        title='%d,%s'%(i,file_name[i])
        # ShowDepthSphere(title=title,setname=setname,depth=depth,Sphere=sphere)
        LabelUtils.ShowDepthSphereMaskHist2(title=title,setname=setname,depth=depth,numPoint=None,jnt_uvd=cur_uvd,hand_points=cur_xyz,Sphere=sphere,partMap=partMap)
        #


        # plt.imshow(ColorMap)
        # plt.savefig("%s%s.jpg"%(lable_dir,file_name[i].split('.')[0]),format='jpg', dpi=300)

        # tmp_name = file_name[i].split('.')[0]
        # img =Image.fromarray(partMap,mode='I')
        # img.save("%s/%s.png"%(lable_dir,tmp_name))
        #
        # img =Image.fromarray(ColorMap, mode='RGB')
        # img.thumbnail((640,480))
        # img.save("%s_color/%s.jpg"%(lable_dir,tmp_name),"JPEG", quality=80)


if __name__=='__main__':

    getPart_mega()
        # labelRefPose_mega(dataset=set)



