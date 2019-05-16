__author__ = 'QiYE'

import numpy

# import matplotlib.pyplot as plt
# import csv
# from src.utils.transformations import *
# from prepare_gen_data_scale import scale_skeleton

# from show_3D_skeleton import show_hand_skeleton
from src.SphereHandModel.utils.transformations import affine_matrix_from_points,decompose_matrix,\
    euler_matrix,quaternion_from_euler,quaternion_matrix,rotation_matrix,compose_matrix

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

def skeleton2sphere_old(Skeleton, initSphere,ratioForPalm,ratioForFingerSphere,palmBaseTopScaleRatio):

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


