__author__ = 'QiYE'

from PIL import Image

from src.SphereHandModel.SkeletonSphereMotion import *
from src.SphereHandModel.ShowSamples import *
from src.SphereHandModel.utils import xyz_uvd


def getCustomizedSphereModel(initSkeleton,initShpere,refSkeleton,scaleInitSphereFactor,palmBaseTopScaleRatio):

    initShpere[4,:]*=scaleInitSphereFactor

    ratioForFingerSphere = get_inter_ratio_bw_jointsphere(initShpere)

    # mot,scale,shear = get_scale_factor(refSkeleton=initSkeleton.copy(),Skeleton=refSkeleton)

    newSphere = initShpere.copy()
    newSphere[1:4,SLICE_IDX_SKE_FROM_SPHERE]= refSkeleton
    ###scale sphere radius####
    # newSphere[4,PALM_SPHERE+SADDLE_SPHERE]*=scale[0][0]
    # figSphereList = [THUMB_SPHERE,INDEX_SPHERE,MIDDLE_SPHERE,RING_SPHERE,PINKY_SPHERE]
    # for i,idx in enumerate(figSphereList):
    #     newSphere[4,idx[1:]]*=scale[i+1][0]
    #
    # print 'initSphere radius',initShpere[4,:]
    # print 'customizedSphere radius', newSphere[4,:]

    ###make sphere model using the palm model from msra cvpr2014's palm model by estimating the affine transformation
    ### and finger model using interploration between the joint locations from current dataset
    ### the location of sphere bw the joint sphere uses the same ratio from msra cvpr2014

    fingerIdxList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    sphereIdxList =[THUMB_INTER_SPHERE,INDEX_INTER_SPHERE,MIDDLE_INTER_SPHERE,RING_INTER_SPHERE,PINKY_INTER_SPHERE]
    numIdx = [0,1,2,3,4]

    for skeIdx,sphIdx,j in zip(fingerIdxList,sphereIdxList,numIdx):
        for i in xrange(len(skeIdx)-1):
              newSphere[1:4,sphIdx[i]]=(refSkeleton[:,skeIdx[i+1]]-refSkeleton[:,skeIdx[i]])*ratioForFingerSphere[j,i]+refSkeleton[:,skeIdx[i]]

    idx=[5,9,13,17]
    scaleMat = numpy.eye(3)*palmBaseTopScaleRatio
    basePalmJoints = numpy.dot(scaleMat,refSkeleton[:,idx]) + numpy.dot(refSkeleton[:,0].reshape(3,1),numpy.ones((1,4)))
    idx = [44,45,46,47]
    newSphere[1:4,idx]=basePalmJoints

    ratioForPalm = get_inter_palm_ratio(initShpere)
    BASE_SPHER = [44,45,46,47]
    TOP_SPHERE=[32,35,38,41]
    num=range(0,4,1)
    for i,b,t in zip(num,BASE_SPHER,TOP_SPHERE):
        newSphere[1:4,t+2]= (newSphere[1:4,t]-newSphere[1:4,b])*ratioForPalm[i,0]+newSphere[1:4,b]
        newSphere[1:4,t+1]= (newSphere[1:4,t]-newSphere[1:4,b])*ratioForPalm[i,1]+newSphere[1:4,b]
    newSphere[1:5,31]=newSphere[1:5,30]


    return newSphere,ratioForPalm,ratioForFingerSphere

def getHandToBeCustomizedFromMsrcFormat_icvl(frameIdx,setname,dataset):
    if setname == 'icvl':
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
    else:
        exit('to be done')

    idx_21 =numpy.array([1,2,3,4,17,5,6,7,18,8,9,10,19,11,12,13,20,14,15,16,21])-1

    xyz.shape = (xyz.shape[0],22,3)
    xyz_21jnt = xyz[:,idx_21,:]*1000
    uvd = xyz_uvd.xyz2uvd(setname=setname, xyz=xyz_21jnt)
    return uvd[frameIdx], xyz_21jnt[frameIdx]


def getInitSphere():
    """use the msra cvpr 2014 hand model's radius to get a rough estimate of
    the sphere radius and the location of inter sphere bw joint sphere"""
    initShpere = numpy.loadtxt('F:/Proj_src/sx/handmodel/msra_handtracking_database/Release_2014_5_28/Subject1/sphere.txt',skiprows=1)[0]
    return initShpere

def customizeSphereHandModel(skeleton,setname):
    initShpere=getInitSphere()
    """move the middle base joint to the origin """
    skeleton-=skeleton[9,:].reshape(1,3)
    initShpere.shape=(48,5)
    initShpere[:,1:4] -= initShpere[35,1:4].reshape(1,3)

    initSkeleton = sphere2skeleton(initShpere)[:,1:4]
    initSkeleton = initSkeleton.T
    initShpere = initShpere.T
    scaleInitSphereFactor=0.8
    palmBaseTopScaleRatio=0.6

    scale_initSphere,ratioForPalm,ratioForFingerSphere = getCustomizedSphereModel(initSkeleton=initSkeleton,initShpere=initShpere,
                                                         refSkeleton=skeleton.T,scaleInitSphereFactor=scaleInitSphereFactor,
                                                         palmBaseTopScaleRatio=palmBaseTopScaleRatio)
    numpy.save('%s_hand_sphere_model.npy'%setname,[scale_initSphere,initShpere,ratioForPalm,
                                                   ratioForFingerSphere,scaleInitSphereFactor,palmBaseTopScaleRatio])




def showCustomizedSphereModelInPointCloud(skeleton,setname):

    Spheres=numpy.load('%s_hand_sphere_model.npy'%setname)
    newSphere=Spheres[0].T
    if setname=='icvl':
        dataset_dir = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/ICVL_dataset_v2_msrc_format/test/depth/'
        file_name = "depth_1_%07d.png" % (1)
        depth = Image.open('%s%s' % (dataset_dir, file_name))
        depth = numpy.asarray(depth, dtype='uint16')
    else:
        exit('To be done')

    newSphere[:,1:4]+=skeleton[9,:].reshape(1,3)

    # initSphere=Spheres[1].T
    # newSphere[:,4]=initSphere[:,4]
    #
    ShowPointCloudFromDepth(setname=setname,depth=depth,hand_points=skeleton,Sphere=newSphere)

def customize_mega():
    setname='mega'
    imgIdx=10
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
    xyz_jnt_gt=[]
    file_name = []

    with open('F:/HuaweiProj/Proj/data/Yang/data/Testing_Yang.txt', 'rb') as f:
        for line in f:
            part = line.split('\t')
            file_name.append(part[0].replace('\\', '/'))
            xyz_jnt_gt.append(part[1:64])

    f.close()

    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float32')

    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]

    skeleton=xyz_jnt_gt[imgIdx].copy()
    initShpere=getInitSphere()
    """move the middle base joint to the origin """
    skeleton-=skeleton[9,:].reshape(1,3)
    initShpere.shape=(48,5)
    initShpere[:,1:4] -= initShpere[35,1:4].reshape(1,3)

    initSkeleton = sphere2skeleton(initShpere)[:,1:4]
    initSkeleton = initSkeleton.T
    initShpere = initShpere.T
    scaleInitSphereFactor=1
    palmBaseTopScaleRatio=1

    scale_initSphere,ratioForPalm,ratioForFingerSphere = getCustomizedSphereModel(initSkeleton=initSkeleton,initShpere=initShpere,
                                                         refSkeleton=skeleton.T,scaleInitSphereFactor=scaleInitSphereFactor,
                                                         palmBaseTopScaleRatio=palmBaseTopScaleRatio)
    numpy.save('%s_hand_sphere_model2.npy'%setname,[scale_initSphere,initShpere,ratioForPalm,
                                                   ratioForFingerSphere,scaleInitSphereFactor,palmBaseTopScaleRatio])

    Spheres=numpy.load('%s_hand_sphere_model2.npy'%setname)
    newSphere=Spheres[0].T
    depth = Image.open('F:/HuaweiProj/Proj/data/%s' % (file_name[imgIdx]))
    depth = numpy.asarray(depth, dtype='uint16')

    newSphere[:,1:4]+=xyz_jnt_gt[imgIdx,9,:].reshape(1,3)

    # initSphere=Spheres[1].T
    # newSphere[:,4]=initSphere[:,4]
    #
    ShowPointCloudFromDepth(setname=setname,depth=depth,hand_points=xyz_jnt_gt[imgIdx,:,:].reshape(21,3),Sphere=newSphere)


if __name__=='__main__':
    customize_mega()
    # """icvl testset"""
    # setname='icvl'
    # dataset='test'
    # _,skeleton = getHandToBeCustomizedFromMsrcFormat_icvl(frameIdx=0,setname=setname,dataset=dataset)
    # customizeSphereHandModel(skeleton=skeleton.copy(),setname=setname)
    # showCustomizedSphereModelInPointCloud(skeleton=skeleton.copy(),setname=setname)