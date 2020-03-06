__author__ = 'QiYE'

import numpy

# import matplotlib.pyplot as plt
# import csv
# from src.utils.transformations import *
# from prepare_gen_data_scale import scale_skeleton

# from show_3D_skeleton import show_hand_skeleton
from src.SphereHandModel.utils.transformations import affine_matrix_from_points,decompose_matrix,\
    euler_matrix,quaternion_from_euler,quaternion_matrix,rotation_matrix,compose_matrix

import h5py

WRIST=0
PALM=[1,5,9,13,17]
GloblaJoint=[0,5,9,13,17]
MCP=[2,6,10,14,18]
DIP=[3,7,11,15,19]
TIP=[4,8,12,16,20]

THUMB=[1,2,3,4]
INDEX=[5,6,7,8]
MIDDLE=[9,10,11,12]
RING=[13,14,15,16]
PINKY=[17,18,19,20]

globIdx4QuatMot = [0,1,2,3]
mcpIdx4QuatMot=[[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
dipIdx4QuatMot=[[24],[26],[28],[30],[32]]
tipIdx4QuatMot=[[25],[27],[29],[31],[33]]

thumbMot = [4,5,6,7,24,25]
indexMot = [8,9,10,11,26,27]
midMot = [12,13,14,15,28,29]
ringMot = [16,17,18,19,30,31]
pinkMot = [20,21,22,23,32,33]


THUMB_JOINT_SPHERE = [30,0,2,4]
THUMB_INTER_SPHERE = [1,3,5]
THUMB_SPHERE = [30,1,0,3,2,5,4]

INDEX_JOINT_SPHERE = [32,6,8,10]
INDEX_INTER_SPHERE = [7,9,11]
INDEX_SPHERE = [32,7,6,9,8,11,10]

MIDDLE_JOINT_SPHERE =[35,12,14,16]
MIDDLE_INTER_SPHERE=[13,15,17]
MIDDLE_SPHERE = [35,13,12,15,14,17,16]

RING_JOINT_SPHERE = [38,18,20,22]
RING_INTER_SPHERE =[19,21,23]
RING_SPHERE = [38,19,18,21,20,23,22]

PINKY_JOINT_SPHERE = [41,24,26,28]
PINKY_INTER_SPHERE = [25,27,29]
PINKY_SPHERE = [41,25,24,27,26,29,28]

WRIST_JOINT_SPHERE = [45]
FINGER_SPHERE = [30,1,0,3,2,5,4,32,7,6,9,8,11,10,35,13,12,15,14,17,16,38,19,18,21,20,23,22,41,25,24,27,26,29,28]
PALM_SPHERE = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
SADDLE_SPHERE = [31]
FINGER_BASE_SPHERE = [30,32,35,38,41]

SLICE_IDX_SKE_FROM_SPHERE=[45,30,0,2,4,32,6,8,10,35,12,14,16,38,18,20,22,41,24,26,28]


def abs2off(label):
    new_label=numpy.empty_like(label)
    new_label[:,WRIST] = label[:,WRIST]
    new_label[:,PALM]=label[:,PALM]
    new_label[:,MCP]=label[:,MCP]-label[:,PALM]
    new_label[:,DIP]=label[:,DIP]-label[:,MCP]
    new_label[:,TIP]=label[:,TIP]-label[:,DIP]
    return new_label



def getFingerSkeleton(FINGER,refSkeleton,fingerMot):

    refOffset = abs2off(refSkeleton)

    b = refOffset[:,FINGER[-1]]
    a =  refOffset[:,FINGER[-2]]
    c=numpy.cross(a,b)
    tmp = numpy.linalg.norm(c)
    dir = c/tmp
    offFinger_3_flexion = fingerMot[-1]
    rotMat34 = rotation_matrix(offFinger_3_flexion,dir)[0:3,0:3]

    b = refOffset[:,FINGER[-2]]
    a =  refOffset[:,FINGER[-3]]
    c=numpy.cross(a,b)
    tmp = numpy.linalg.norm(c)
    dir = c/tmp
    offFinger_2_flexion = fingerMot[-2]
    rotMat23 = rotation_matrix(offFinger_2_flexion,dir)[0:3,0:3]

    refFingerJoint = numpy.zeros((3,4))
    refFingerJoint[:,-3] = refOffset[:,FINGER[-3]]
    refFingerJoint[:,-2] = numpy.dot(rotMat23 , refOffset[:,FINGER[-2]]) + refFingerJoint[:,-3]
    refFingerJoint[:,-1] = numpy.dot(rotMat23 , numpy.dot(rotMat34, refOffset[:,FINGER[-1]])) + refFingerJoint[:,-2]

    rotMat = quaternion_matrix(fingerMot[0:4])[0:3,0:3]

    return numpy.dot(rotMat, refFingerJoint)

def norm_quat_from_pred_mot(motion):
    loc = numpy.where(motion[globIdx4QuatMot][0]<0)

    motion[globIdx4QuatMot][0][loc]=-motion[globIdx4QuatMot][0][loc]
    n = numpy.sqrt( numpy.sum( motion[globIdx4QuatMot]**2,axis=0))
    motion[globIdx4QuatMot] /= n

    mcpIdx4QuatMot=[[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23]]
    for fingQuatIdx in  mcpIdx4QuatMot:
        loc = numpy.where(motion[fingQuatIdx][0]<0)
        motion[fingQuatIdx][0][loc]=-motion[fingQuatIdx][0][loc]
        n = numpy.sqrt( numpy.sum( motion[fingQuatIdx]**2,axis=0))
        motion[fingQuatIdx] /=n

    return



def motion2skeleton(refSkeleton,motion):
    Skeleton= refSkeleton.copy()
    figSkeList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    figMotList = [thumbMot,indexMot,midMot,ringMot,pinkMot]

    globRotMat = quaternion_matrix(motion[globIdx4QuatMot])[0:3,0:3]

    for figMotIdx,figSkeIdx in zip(figMotList,figSkeList):
        tmp = getFingerSkeleton(FINGER=figSkeIdx,refSkeleton=refSkeleton,fingerMot=motion[figMotIdx])
        Skeleton[:,figSkeIdx[1:4]]= tmp[:,1:4] + numpy.dot(Skeleton[:,figSkeIdx[0]].reshape(3,1),numpy.ones((1,3)))
    Skeleton = numpy.dot(globRotMat, Skeleton)
    return Skeleton


def getFingerBoneRot(FINGER,refSkeleton,reverSkeleton,shear=False, scale=False, usesvd=True):
    """
    each column is a joint location"""
    offset = abs2off(reverSkeleton)
    refOffset = abs2off(refSkeleton)

    Motion=[]
    b = offset[:,FINGER[-1]]
    a =  offset[:,FINGER[-2]]
    c=numpy.cross(a,b)
    tmp = numpy.linalg.norm(c)

    Finger_3_flexion = numpy.arctan2(tmp,numpy.dot(a,b))
    b = refOffset[:,FINGER[-1]]
    a =  refOffset[:,FINGER[-2]]
    c=numpy.cross(a,b)
    tmp = numpy.linalg.norm(c)
    dir = c/tmp

    refFinger_3_flexion = numpy.arctan2(tmp,numpy.dot(a,b))
    offFinger_3_flexion = Finger_3_flexion-refFinger_3_flexion
    # print offFinger_3_flexion
    rotMat34 = rotation_matrix(offFinger_3_flexion,dir)[0:3,0:3]

    b = offset[:,FINGER[-2]]
    a = offset[:,FINGER[-3]]
    c=numpy.cross(a,b)
    tmp = numpy.linalg.norm(c)
    Finger_2_flexion = numpy.arctan2(tmp,numpy.dot(a,b))

    b = refOffset[:,FINGER[-2]]
    a =  refOffset[:,FINGER[-3]]
    c=numpy.cross(a,b)
    tmp = numpy.linalg.norm(c)
    dir = c/tmp
    # print 'dot of dir23,dir34', numpy.dot(tmp_dir,dir)
    refFinger_2_flexion = numpy.arctan2(tmp,numpy.dot(a,b))
    offFinger_2_flexion = Finger_2_flexion-refFinger_2_flexion
    rotMat23 = rotation_matrix(offFinger_2_flexion,dir)[0:3,0:3]


    refFingerJoint = numpy.zeros((3,4))
    refFingerJoint[:,-3] = refOffset[:,FINGER[-3]]
    refFingerJoint[:,-2] = numpy.dot(rotMat23 , refOffset[:,FINGER[-2]]) + refFingerJoint[:,-3]
    refFingerJoint[:,-1] = numpy.dot(rotMat23 , numpy.dot(rotMat34, refOffset[:,FINGER[-1]])) + refFingerJoint[:,-2]

    reverFingerJoint = numpy.zeros((3,4))
    reverFingerJoint[:,1:4]=reverSkeleton[:,FINGER[1:4]]- numpy.dot(reverSkeleton[:,FINGER[0]].reshape(3,1),numpy.ones((1,3)))

    affinematrix = affine_matrix_from_points(v0=refFingerJoint, v1=reverFingerJoint, shear=shear, scale=scale, usesvd=usesvd)
    scale0, shear0, angles0, trans0, persp0 = decompose_matrix(affinematrix)
    if scale == True:
        print 'finger scale0',scale0,shear0
    if shear == True:
        print 'finger shear0',shear0
    # print ' angles, trans,', angles, trans
    # rotMat = euler_matrix(*angles, axes='sxyz')[0:3,0:3]
    # tmp = numpy.dot(rotMat, refFingerJoint) + numpy.dot(trans.reshape(3,1),numpy.ones((1,4)))
    # print numpy.allclose(tmp, reverFingerJoint)
    Motion+=angles0
    Motion+=[offFinger_2_flexion]
    Motion+=[offFinger_3_flexion]
    # err = numpy.sqrt(numpy.sum((tmp-reverFingerJoint)**2,axis=0))
    # print numpy.mean(err)
    # print err
    # return Motion,trans

    return Motion,scale0,shear0



def skeleton2motion(refSkeleton,Skeleton,shear=False, scale=True, usesvd=True):

    quatMotion = numpy.zeros((34,))
    numJoint=21
    PalmJoints = Skeleton[:,PALM]
    refPalmJoints = refSkeleton[:,PALM]
    affinematrix = affine_matrix_from_points(v0=refPalmJoints, v1=PalmJoints, shear=shear, scale=scale, usesvd=usesvd)
    scale0, shear0, angles0, trans0, persp0 = decompose_matrix(affinematrix)
    print 'palm', scale0
    q = quaternion_from_euler(*angles0,axes='sxyz')
    if q[0]<0:
        q[0]=-q[0]
    quatMotion[globIdx4QuatMot] = q


    if scale==False:
        rotMat = euler_matrix(*angles0, axes='sxyz')[0:3,0:3]
        reverSkeleton = numpy.dot((rotMat).T, Skeleton - numpy.dot(trans0.reshape(3,1),numpy.ones((1,numJoint))))
    else:
        rotMat = compose_matrix(scale=scale0, shear=None, angles=angles0, translate=None,perspective=None)[0:3,0:3]
        reverSkeleton = numpy.dot((rotMat).T, Skeleton - numpy.dot(trans0.reshape(3,1),numpy.ones((1,numJoint))))

    figSkeList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    figMotList = [thumbMot,indexMot,midMot,ringMot,pinkMot]
    for figMotIdx,figSkeIdx in zip(figMotList,figSkeList):
        tmp = getFingerBoneRot(FINGER=figSkeIdx,refSkeleton=refSkeleton,reverSkeleton=reverSkeleton,shear=shear, scale=scale, usesvd=usesvd)
        q = quaternion_from_euler(tmp[0],tmp[1],tmp[2],axes='sxyz')
        if q[0]<0:
            q[0]=-q[0]
        quatMotion[figMotIdx[0:4]]=q
        quatMotion[figMotIdx[4:6]]=tmp[3:5]

    return quatMotion

"""
usage
refSkeleton = (refSkeleton-refSkeleton[9,:]).T
mean_scale = numpy.zeros((6,3))
for i in numpy.random.randint(0,num_ori_sample,50):
    mot,scale = get_scale_factor(refSkeleton=refSkeleton.copy(),Skeleton=norm_xyz[:,:,i],shear=False, scale=True, usesvd=True)
    mean_scale+=scale
print mean_scale/50
scale_skeleton = scale_skeleton(mean_scale/50,refSkeleton)


mean_scale = numpy.zeros((6,3))
for i in numpy.random.randint(0,num_ori_sample,50):
    mot,scale = get_scale_factor(refSkeleton=scale_skeleton,Skeleton=norm_xyz[:,:,i],shear=False, scale=True, usesvd=True)
    mean_scale+=scale
print mean_scale/50

"""
def evaluate_scale_factor(refSkeleton,Skeleton,shear_palm=False, shear_finger=False,scale=True, usesvd=True):
    scale_factor = []
    shear_factor= []
    quatMotion = numpy.zeros((34,))
    numJoint=21
    PalmJoints = Skeleton[:,GloblaJoint]
    refPalmJoints = refSkeleton[:,GloblaJoint]
    affinematrix = affine_matrix_from_points(v0=refPalmJoints, v1=PalmJoints, shear=shear_palm, scale=scale, usesvd=usesvd)
    scale0, shear0, angles0, trans0, persp0 = decompose_matrix(affinematrix)
    print 'palm', scale0,shear0,trans0
    scale_factor.append(scale0)
    shear_factor.append(shear0)
    # mat= compose_matrix(scale=scale0, shear=shear0, angles=angles0, translate=None,perspective=None)[0:3,0:3]
    # # mat= compose_matrix(scale=None, shear=None, angles=angles0, translate=None,perspective=None)[0:3,0:3]
    # newrefpalm = numpy.dot(mat, refSkeleton)
    # show_hand_skeleton(refSkeleton=refSkeleton.T,Skeleton=Skeleton.T, tranSkeleton=newrefpalm.T)


    q = quaternion_from_euler(*angles0,axes='sxyz')
    if q[0]<0:
        q[0]=-q[0]
    quatMotion[globIdx4QuatMot] = q


    rotMat= compose_matrix(scale=None, shear=None, angles=angles0, translate=None,perspective=None)[0:3,0:3]
    reverSkeleton = numpy.dot((rotMat).T, Skeleton - numpy.dot(trans0.reshape(3,1),numpy.ones((1,numJoint))))
    # show_hand_skeleton(refSkeleton=refSkeleton.T,Skeleton=Skeleton.T, tranSkeleton=reverSkeleton.T)


    figSkeList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    figMotList = [thumbMot,indexMot,midMot,ringMot,pinkMot]
    for figMotIdx,figSkeIdx in zip(figMotList,figSkeList):
        tmp,fingScale,fingShear = getFingerBoneRot(FINGER=figSkeIdx,refSkeleton=refSkeleton,
                                                   reverSkeleton=reverSkeleton,shear=shear_finger, scale=scale, usesvd=usesvd)
        scale_factor.append(fingScale)
        shear_factor.append(fingShear)
        q = quaternion_from_euler(tmp[0],tmp[1],tmp[2],axes='sxyz')
        if q[0]<0:
            q[0]=-q[0]
        quatMotion[figMotIdx[0:4]]=q
        quatMotion[figMotIdx[4:6]]=tmp[3:5]

    return quatMotion, scale_factor,shear_factor


def get_scale_factor(refSkeleton,Skeleton):
    scale_factor = []
    shear_factor= []
    quatMotion = numpy.zeros((34,))
    numJoint=21
    PalmJoints = Skeleton[:,GloblaJoint]
    refPalmJoints = refSkeleton[:,GloblaJoint]
    affinematrix = affine_matrix_from_points(v0=refPalmJoints, v1=PalmJoints, shear=False, scale=True, usesvd=True)
    scale0, shear0, angles0, trans0, persp0 = decompose_matrix(affinematrix)
    print 'palm,scale0,shear0,trans0', scale0,shear0,trans0
    scale_factor.append(scale0)
    shear_factor.append(shear0)

    newrefSkeleton = refSkeleton.copy()
    scaleMat = compose_matrix(scale=scale0, shear=shear0, angles=None, translate=trans0,perspective=None)[0:3,0:3]
    newrefSkeleton[:,GloblaJoint] =  numpy.dot(scaleMat, newrefSkeleton[:,GloblaJoint])

    figSkeList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    for i,idx in enumerate(figSkeList):
        newrefSkeleton[:,idx[1:]] = refSkeleton[:,idx[1:]] - refSkeleton[:,idx[0]].reshape(3,1)
        newrefSkeleton[:,idx[1:]] += newrefSkeleton[:,idx[0]].reshape(3,1)

    # show_hand_skeleton(refSkeleton=refSkeleton.T,Skeleton=Skeleton.T, tranSkeleton=newrefSkeleton.T)


    PalmJoints = Skeleton[:,GloblaJoint]
    refPalmJoints = newrefSkeleton[:,GloblaJoint]
    affinematrix = affine_matrix_from_points(v0=refPalmJoints, v1=PalmJoints, shear=False, scale=False, usesvd=True)
    scale0, shear0, angles0, trans0, persp0 = decompose_matrix(affinematrix)

    print 'palm scale0,shear0,trans0', scale0,shear0,trans0
    q = quaternion_from_euler(*angles0,axes='sxyz')
    if q[0]<0:
        q[0]=-q[0]
    quatMotion[globIdx4QuatMot] = q

    # scaleMat = compose_matrix(scale=None, shear=None, angles=angles0, translate=trans0,perspective=None)[0:3,0:3]
    # newrefSkeleton2=  numpy.dot(scaleMat, newrefSkeleton)
    # show_hand_skeleton(refSkeleton=newrefSkeleton.T,Skeleton=Skeleton.T, tranSkeleton=newrefSkeleton2.T)
    rotMat = compose_matrix(scale=None, shear=None, angles=angles0, translate=None,perspective=None)[0:3,0:3]
    reverSkeleton = numpy.dot((rotMat).T, Skeleton - numpy.dot(trans0.reshape(3,1),numpy.ones((1,numJoint))))
    # show_hand_skeleton(refSkeleton=refSkeleton.T,Skeleton=newrefSkeleton.T, tranSkeleton=reverSkeleton.T)


    figSkeList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    figMotList = [thumbMot,indexMot,midMot,ringMot,pinkMot]

    for figMotIdx,figSkeIdx in zip(figMotList,figSkeList):
        tmp,fingScale,fingShear = getFingerBoneRot(FINGER=figSkeIdx,refSkeleton=newrefSkeleton,reverSkeleton=reverSkeleton,
                                                   shear=False, scale=True, usesvd=True)
        scale_factor.append(fingScale)
        shear_factor.append(fingShear)
        q = quaternion_from_euler(tmp[0],tmp[1],tmp[2],axes='sxyz')
        if q[0]<0:
            q[0]=-q[0]
        quatMotion[figMotIdx[0:4]]=q
        quatMotion[figMotIdx[4:6]]=tmp[3:5]

    # show_hand_skeleton(refSkeleton=newrefSkeleton.T,Skeleton=reverSkeleton.T, tranSkeleton=standSkeleton.T)
    return quatMotion, scale_factor,shear_factor


def get_scale_skeleton(scale,shear,skeleton):
    newSkeleton = skeleton.copy()
    scaleMat= compose_matrix(scale=scale[0], shear=shear[0], angles=None, translate=None,perspective=None)[0:3,0:3]
    newSkeleton[:,GloblaJoint] =  numpy.dot(scaleMat, newSkeleton[:,GloblaJoint])

    figSkeList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    for i,idx in enumerate(figSkeList):
        newSkeleton[:,idx[1:]] -= newSkeleton[:,idx[0]].reshape(3,1)
        scaleMat= compose_matrix(scale=scale[i+1], shear=shear[i+1], angles=None, translate=None,perspective=None)[0:3,0:3]
        newSkeleton[:,idx[1:]] = numpy.dot(scaleMat,newSkeleton[:,idx[1:]])
        newSkeleton[:,idx[1:]] += newSkeleton[:,idx[0]].reshape(3,1)
    return newSkeleton

def get_scale_sphere(scale,shear,sphere):
    newSphere = sphere.copy()

    scaleMat= compose_matrix(scale=scale[0], shear=shear, angles=None, translate=None,perspective=None)[0:3,0:3]
    ###scale xyz cooridnate####
    newSphere[1:4,PALM_SPHERE+SADDLE_SPHERE] = numpy.dot(scaleMat, newSphere[1:4,PALM_SPHERE+SADDLE_SPHERE])
    ###scale radius####
    newSphere[4,PALM_SPHERE]*=scale[0,0]

    figSphereList = [THUMB_SPHERE,INDEX_SPHERE,MIDDLE_SPHERE,RING_SPHERE,PINKY_SPHERE]

    for i,idx in enumerate(figSphereList):
        newSphere[1:4,idx[1:]] = newSphere[1:4,idx[1:]]- newSphere[1:4,idx[0]].reshape(3,1)

        scaleMat= compose_matrix(scale=scale[i+1], shear=None, angles=None, translate=None,perspective=None)[0:3,0:3]
        newSphere[1:4,idx[1:]] = numpy.dot(scaleMat,newSphere[1:4,idx[1:]])
        newSphere[4,idx[1:]]*=scale[i+1,0]

        newSphere[1:4,idx[1:]] = newSphere[1:4,idx[1:]] + newSphere[1:4,idx[0]].reshape(3,1)

    return newSphere

def get_inter_ratio_bw_jointsphere(shpere):

    ratioBWsphere  = numpy.empty((5,3))

    figSphereList = [THUMB_SPHERE,INDEX_SPHERE,MIDDLE_SPHERE,RING_SPHERE,PINKY_SPHERE]
    for i,idx in enumerate(figSphereList):
        for j in [0,2,4]:
            ratioBWsphere[i,j/2] = (shpere[4,idx[j]]+shpere[4,idx[j+1]])/(shpere[4,idx[j]]+shpere[4,idx[j+2]]+2*shpere[4,idx[j+1]])
    return ratioBWsphere

def get_inter_palm_ratio(sphere):

    ratioBWsphere  = numpy.empty((4,2))
    BASE_SPHER = [44,45,46,47]
    TOP_SPHERE=[32,35,38,41]
    num=range(0,4,1)
    for i,b,t in zip(num,BASE_SPHER,TOP_SPHERE):
        boneLen = sphere[4,t]+sphere[4,t+1]*2+sphere[4,t+2]*2+sphere[4,b]
        ratioBWsphere[i,0] = (sphere[4,b] +sphere[4,t+2])/boneLen
        ratioBWsphere[i,1] = (sphere[4,t+1]+sphere[4,t+2]*2+sphere[4,b])/boneLen
    return ratioBWsphere



def skeleton2sphere_old(structSke, initSphere,motFromInitSphere,ratioForFingerSphere):

    sphere = numpy.empty((48,5))
    sphere[1:4,SLICE_IDX_SKE_FROM_SPHERE]=structSke
    """TO DO"""
    """THUMB"""

    fingerIdxList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    sphereIdxList =[THUMB_INTER_SPHERE,INDEX_INTER_SPHERE,MIDDLE_INTER_SPHERE,RING_INTER_SPHERE,PINKY_INTER_SPHERE]
    numIdx = [0,1,2,3,4]

    for skeIdx,sphIdx,j in zip(fingerIdxList,sphereIdxList,numIdx):
        for i in xrange(len(skeIdx)-1):
              sphere[sphIdx[i]]=(structSke[skeIdx[i+1]]-structSke[skeIdx[i]])*ratioForFingerSphere[j,i]+structSke[skeIdx[i]]

    globRotMat = quaternion_matrix(motFromInitSphere[globIdx4QuatMot])[0:3,0:3]
    sphere[PALM_SPHERE+SADDLE_SPHERE] = numpy.dot(globRotMat, initSphere[PALM_SPHERE+SADDLE_SPHERE])


    return sphere

def skeleton2sphere(Skeleton, initSphere,ratioForPalm,ratioForFingerSphere,palmBaseTopScaleRatio):

    newSphere = initSphere.copy()
    newSphere[1:4,SLICE_IDX_SKE_FROM_SPHERE]=Skeleton.copy()

    fingerIdxList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    sphereIdxList =[THUMB_INTER_SPHERE,INDEX_INTER_SPHERE,MIDDLE_INTER_SPHERE,RING_INTER_SPHERE,PINKY_INTER_SPHERE]
    numIdx = [0,1,2,3,4]

    for skeIdx,sphIdx,j in zip(fingerIdxList,sphereIdxList,numIdx):
        for i in xrange(len(skeIdx)-1):
              newSphere[1:4,sphIdx[i]]=(Skeleton[:,skeIdx[i+1]]-Skeleton[:,skeIdx[i]])*ratioForFingerSphere[j,i]+Skeleton[:,skeIdx[i]]

    idx=[5,9,13,17]
    scaleMat = numpy.eye(3)*palmBaseTopScaleRatio
    tmpSke = Skeleton-Skeleton[:,9].reshape(3,1)
    basePalmJoints = numpy.dot(scaleMat,tmpSke[:,idx]) + numpy.dot(tmpSke[:,0].reshape(3,1),numpy.ones((1,4)))+Skeleton[:,9].reshape(3,1)
    idx = [44,45,46,47]
    newSphere[1:4,idx]=basePalmJoints

    BASE_SPHER = [44,45,46,47]
    TOP_SPHERE=[32,35,38,41]
    num=range(0,4,1)
    for i,b,t in zip(num,BASE_SPHER,TOP_SPHERE):
        newSphere[1:4,t+2]= (newSphere[1:4,t]-newSphere[1:4,b])*ratioForPalm[i,0]+newSphere[1:4,b]
        newSphere[1:4,t+1]= (newSphere[1:4,t]-newSphere[1:4,b])*ratioForPalm[i,1]+newSphere[1:4,b]
    newSphere[1:,31]=newSphere[1:,30]

    return newSphere




def sphere2skeleton(sphere):
    return sphere[SLICE_IDX_SKE_FROM_SPHERE,:]




def drawStructHandSamplesFromGtMotionUniform(rng,gaussParams,NumSamples,isDTipCorrelated):
    batch_size = gaussParams.shape[-1]
    # if NumSamples==1:
    #     return gaussParams[0:34]
    newMotions=numpy.zeros((34,NumSamples,batch_size))
    dg = rng.unifrom(4,6,NumSamples,batch_size)
    n = numpy.sqrt( numpy.sum( dg**2, axis=0))
    d = dg/n
    palmmcpMeanParams = numpy.reshape(gaussParams[0:24,:], (4,6,1,batch_size),order='F')
    palmmcpDivParams = numpy.reshape(gaussParams[34:58,:],(4,6,1,batch_size),order='F')
    #when we have the divation estimation, we can draw samples according the divation easily like the funciton drawStructHandSamplesFromGaussParams
    # dn = palmmcpDivParams*d+palmmcpMeanParams
    # when we do not have the range of uncertainy, we sample based on the assumption of the motion value: the larger the motion, the larger the uncertainty is
    # so the diviation is propotinal to the movement range, i.e. the motion value
    dn = palmmcpDivParams*d*palmmcpMeanParams+palmmcpMeanParams

    neg = numpy.where(dn[0] < 0)
    dn[0][neg] = -dn[0][neg]# the range of the rotation t about axis (xyz) is [-pi,pi]
    # tmp  = numpy.sqrt(numpy.sum( InitQuaternion**2, axis=0))
    n2 = numpy.sqrt(numpy.sum( dn**2, axis=0))
    dn2 = dn/n2
    newMotions[0:24] = numpy.reshape(dn2,(24,NumSamples,batch_size),order='F')

    meanDip = gaussParams[24:34:2]
    meanTip = gaussParams[25:34:2]
    divDip = gaussParams[58:68:2]
    divTip = gaussParams[59:68:2]

    divDip.shape= (5,1,batch_size)
    divTip.shape= (5,1,batch_size)
    meanDip.shape= (5,1,batch_size)
    meanTip.shape= (5,1,batch_size)

    dg = rng.randn(2,5,NumSamples,batch_size)
    if isDTipCorrelated == True:
        corDTip = gaussParams[68:73]
        corDTip.shape=(5,1,batch_size)
        dg[1] = dg[0]*corDTip+dg[1]*numpy.sqrt(1-corDTip**2)
    else:
        angleDipRange=[-0.17,2.35]
        angleTipRange=[-0.17,1.58]
        tmp = dg[0]*dg[1]
        loc = numpy.where(tmp<0)
        dg[1][loc] = -dg[1][loc]

        dg[0]=dg[0]*divDip*meanDip+meanDip
        dg[0][numpy.where(dg[0]<angleDipRange[0])]=angleDipRange[0]
        dg[0][numpy.where(dg[0]>angleDipRange[1])]=angleDipRange[1]
        dg[1]=dg[1]*divTip*meanTip+meanTip
        dg[1][numpy.where(dg[1]<angleTipRange[0])]=angleTipRange[0]
        dg[1][numpy.where(dg[1]>angleTipRange[1])]=angleTipRange[1]
    # dg[0]=dg[0]*divDip+meanDip
    # dg[1]=dg[1]*divTip+meanTip
    # dg[0]=dg[0]*divDip*meanDip+meanDip
    # dg[1]=dg[1]*divTip*meanTip+meanTip
    newMotions[24:34] = numpy.reshape(dg,(10,NumSamples,batch_size),order='F')
    return newMotions


def drawStructHandSamplesFromGtMotion(rng,gaussParams,NumSamples,isDTipCorrelated):
    batch_size = gaussParams.shape[-1]
    # if NumSamples==1:
    #     return gaussParams[0:34]
    newMotions=numpy.zeros((34,NumSamples,batch_size))
    dg = rng.randn(4,6,NumSamples,batch_size)
    n = numpy.sqrt( numpy.sum( dg**2, axis=0))
    d = dg/n
    palmmcpMeanParams = numpy.reshape(gaussParams[0:24,:], (4,6,1,batch_size),order='F')
    palmmcpDivParams = numpy.reshape(gaussParams[34:58,:],(4,6,1,batch_size),order='F')
    #when we have the divation estimation, we can draw samples according the divation easily like the funciton drawStructHandSamplesFromGaussParams
    # dn = palmmcpDivParams*d+palmmcpMeanParams
    # when we do not have the range of uncertainy, we sample based on the assumption of the motion value: the larger the motion, the larger the uncertainty is
    # so the diviation is propotinal to the movement range, i.e. the motion value
    dn = palmmcpDivParams*d*palmmcpMeanParams+palmmcpMeanParams

    neg = numpy.where(dn[0] < 0)
    dn[0][neg] = -dn[0][neg]# the range of the rotation t about axis (xyz) is [-pi,pi]
    # tmp  = numpy.sqrt(numpy.sum( InitQuaternion**2, axis=0))
    n2 = numpy.sqrt(numpy.sum( dn**2, axis=0))
    dn2 = dn/n2
    newMotions[0:24] = numpy.reshape(dn2,(24,NumSamples,batch_size),order='F')

    meanDip = gaussParams[24:34:2]
    meanTip = gaussParams[25:34:2]
    divDip = gaussParams[58:68:2]
    divTip = gaussParams[59:68:2]

    divDip.shape= (5,1,batch_size)
    divTip.shape= (5,1,batch_size)
    meanDip.shape= (5,1,batch_size)
    meanTip.shape= (5,1,batch_size)

    dg = rng.randn(2,5,NumSamples,batch_size)
    if isDTipCorrelated == True:
        corDTip = gaussParams[68:73]
        corDTip.shape=(5,1,batch_size)
        dg[1] = dg[0]*corDTip+dg[1]*numpy.sqrt(1-corDTip**2)
    else:
        angleDipRange=[-0.17,2.35]
        angleTipRange=[-0.17,1.58]
        tmp = dg[0]*dg[1]
        loc = numpy.where(tmp<0)
        dg[1][loc] = -dg[1][loc]

        dg[0]=dg[0]*divDip*meanDip+meanDip
        dg[0][numpy.where(dg[0]<angleDipRange[0])]=angleDipRange[0]
        dg[0][numpy.where(dg[0]>angleDipRange[1])]=angleDipRange[1]
        dg[1]=dg[1]*divTip*meanTip+meanTip
        dg[1][numpy.where(dg[1]<angleTipRange[0])]=angleTipRange[0]
        dg[1][numpy.where(dg[1]>angleTipRange[1])]=angleTipRange[1]
    # dg[0]=dg[0]*divDip+meanDip
    # dg[1]=dg[1]*divTip+meanTip
    # dg[0]=dg[0]*divDip*meanDip+meanDip
    # dg[1]=dg[1]*divTip*meanTip+meanTip
    newMotions[24:34] = numpy.reshape(dg,(10,NumSamples,batch_size),order='F')
    return newMotions



def drawStructHandSamplesFromGaussParams(rng,gaussParams,NumSamples,isDTipCorrelated):
    batch_size = gaussParams.shape[-1]
    # if NumSamples==1:
    #     return gaussParams[0:34].reshape(34,1,batch_size)

    newMotions=numpy.zeros((34,NumSamples,batch_size))

    dg = rng.randn(4,6,NumSamples,batch_size)
    n = numpy.sqrt( numpy.sum( dg**2, axis=0))
    d = dg/n
    palmmcpMeanParams = numpy.reshape(gaussParams[0:24,:],(4,6,1,batch_size),order='F')
    palmmcpDivParams = numpy.reshape(gaussParams[34:58,:],(4,6,1,batch_size),order='F')
    dn = palmmcpDivParams*d+palmmcpMeanParams

# dn = (stand_div_rot*d+1)*numpy.dot(InitQuaternion.reshape(4,1),numpy.ones((1, NumSamples)))
    neg = numpy.where(dn[0] < 0)
    dn[0][neg] = -dn[0][neg]# the range of the rotation t about axis (xyz) is [-pi,pi]
    # tmp  = numpy.sqrt(numpy.sum( InitQuaternion**2, axis=0))
    n2 = numpy.sqrt(numpy.sum( dn**2, axis=0))
    dn2 = dn/n2
    newMotions[0:24] = numpy.reshape(dn2,(24,NumSamples,batch_size),order='F')

    meanDip = gaussParams[24:34:2]
    meanTip = gaussParams[25:34:2]
    divDip = gaussParams[58:68:2]
    divTip = gaussParams[59:68:2]

    divDip.shape= (5,1,batch_size)
    divTip.shape= (5,1,batch_size)
    meanDip.shape= (5,1,batch_size)
    meanTip.shape= (5,1,batch_size)

    dg = rng.randn(2,5,NumSamples,batch_size)

    if isDTipCorrelated == True:
        corDTip = gaussParams[68:73]
        corDTip.shape=(5,1,batch_size)
        dg[1] = dg[0]*corDTip+dg[1]*numpy.sqrt(1-corDTip**2)
    else:
        tmp = dg[0]*dg[1]
        loc = numpy.where(tmp<0)
        dg[1][loc] = -dg[1][loc]
        dg[0]=dg[0]*divDip*meanDip+meanDip
        dg[1]=dg[1]*divTip*meanTip+meanTip


    # dg[0]=dg[0]*divDip+meanDip
    # dg[1]=dg[1]*divTip+meanTip
    newMotions[24:34] = numpy.reshape(dg,(10,NumSamples,batch_size),order='F')
    # newMotions[:,0,:]=gaussParams[0:34]
    return newMotions


# def score(NumSamples , gaussParams, gtSkeleton,refSkeleton,eulerDistWeight = 0.001,logDistWeight=1):
#     """all gaussParams, motions and skeletons are stored in column wise"""
#     batch_size = gaussParams.shape[0]
#     rng = numpy.random.RandomState(2**30)
#     motSamples = drawStructHandSamplesFromGaussParams(rng,gaussParams,NumSamples)
#     skeSamples4Next  = numpy.empty((3,21,NumSamples,batch_size))
#     motLabels4Next = numpy.empty_like(motSamples)
#     for i in range(NumSamples):
#         for j in range(batch_size):
#             tmp =motion2skeleton(refSkeleton=refSkeleton[:,j],motion=motSamples[:,i,j])
#             motLabels4Next[:,i,j] = skeleton2motion(refSkeleton=tmp,Skeleton=gtSkeleton[:,:,j])
#             skeSamples4Next[:,:,i,j]=tmp
#
#     score = numpy.mean(distMax*eulerDistWeight+logDistWeight*numpy.log(distMax+0.0001))
#     distMax = numpy.max(numpy.sqrt(numpy.sum((skeSamples4Next-gtSkeleton)**2,axis=0)),axis=-1)
#     # score = numpy.mean(distMax*eulerDistWeight+logDistWeight*numpy.log(distMax+0.0001))
#     return motSamples, skeSamples4Next.reshape(shape=(63,NumSamples,batch_size),order='F'), motLabels4Next,     score



#
# def test_trans_bw_Ske_Mot():
#
#     path = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/NYU_dataset/NYU_dataset/nyu_msrc_format/test/log_file_0_v3.csv'
#     xyz=  numpy.empty((8252,66),dtype='float32')
#     with open(path, 'rb') as f:
#         reader = list(csv.reader(f.read().splitlines()))
#         for i in rang(6,len(reader),1):
#             xyz[i-6]=reader[i][110:176]
#     f.close()
#
#     idx_21 =numpy.array([1,2,3,4,17,5,6,7,18,8,9,10,19,11,12,13,20,14,15,16,21])-1
#     xyz.shape = (xyz.shape[0],22,3)
#     xyz_21jnt = xyz[:,idx_21,:]
#
#     xyz_tmp=(xyz_21jnt-xyz_21jnt[:,9,:].reshape(xyz_21jnt.shape[0],1,3)).T*1000
#     norm_xyz = scale_skeleton(inSkeleton=xyz_tmp)
#
#     refSkeleton=norm_xyz[:,:,0]
#     sumerr =[]
#     for i in range(2441,8252,1):
#     # for i in numpy.random.randint(0,2400,10):
#     # for i in numpy.random.randint(2500,num_ori_sample,10):
#         motion = skeleton2motion(refSkeleton=refSkeleton,Skeleton=norm_xyz[:,:,i])# the motion of ref pose to the gt pose
#         tmp = motion2skeleton(refSkeleton=refSkeleton,motion=motion)
#         # show_hand_skeleton(refSkeleton=refSkeleton.T,Skeleton=norm_xyz[:,:,i].T, tranSkeleton=tmp.T)
#
#         err = numpy.mean(numpy.sqrt(numpy.sum((tmp-norm_xyz[:,:,i])**2,axis=0)))
#         # print(err)
#         sumerr.append(err)
#     # print(numpy.mean(sumerr))
#     return
#

def calculate_scale_factor():
    dataset_path_prefix='F:/Proj_Struct_RL_v2/data'
    f = h5py.File('%s/icvl/source/test_norm_hand_uvd_rootmid_scale_msrcformat.h5'%(dataset_path_prefix), 'r+')
    # r = f['r0'][...]
    refSkeleton = f['refSkeleton'][...]
    xyz_jnt_gt = f['xyz_jnt_gt_ori'][...]
    f.close()
    # print numpy.max(refSkeleton)
    # print numpy.min(refSkeleton)

    num_ori_sample=xyz_jnt_gt.shape[0]
    #bbsizre = 30cm,so half hand size is 0.015m
    norm_xyz = (xyz_jnt_gt - xyz_jnt_gt[:,9,:].reshape(num_ori_sample,1,3)).T

    sumerr =[]
    # for i in range(0,num_ori_sample,1):
    # for i in numpy.random.randint(0,2400,10):
    for i in numpy.random.randint(0,num_ori_sample,100):
        motion = skeleton2motion(refSkeleton=refSkeleton,Skeleton=norm_xyz[:,:,i],shear=False, scale=True, usesvd=True)# the motion of ref pose to the gt pose
        tmp = motion2skeleton(refSkeleton=refSkeleton,motion=motion)
        # show_hand_skeleton(refSkeleton=refSkeleton.T,Skeleton=norm_xyz[:,:,i].T, tranSkeleton=tmp.T)

        err = numpy.mean(numpy.sqrt(numpy.sum((tmp-norm_xyz[:,:,i])**2,axis=0)))
        # print(err)
        sumerr.append(err)
    # print(numpy.mean(sumerr))


#
if __name__=='__main__':

    load_data()

