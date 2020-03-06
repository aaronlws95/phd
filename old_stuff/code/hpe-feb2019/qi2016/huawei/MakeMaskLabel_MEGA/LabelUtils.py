__author__ = 'QiYE'

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy
from mpl_toolkits.mplot3d import Axes3D
from . import xyz_uvd

#####Hand Model Parameter for msrc test######

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
facecolor = numpy.random.rand(21,3)
# facecolor2 = numpy.random.rand(19,3)
# facecolor2 = numpy.array([[ 1,1,1],
#        [ 0.2854957 ,  0.998758  ,  0.25251318],
#        [ 0.48558226,  0.13333592,  0.78539616],
#        [ 0.61417163,  0.94196445,  0.88943005],
#        [ 0.45937801,  0.14810284,  0.13976017],
#        [ 0.52574357,  0.58866709,  0.873374  ],
#        [ 0.77529657,  0.8954228 ,  0.16515325],
#        [ 0.15040765,  0.71614595,  0.31375659],
#        [ 0.26094682,  0.49756599,  0.79440151],
#        [ 0.41102997,  0.05214649,  0.54266583],
#        [ 0.18477406,  0.72949156,  0.85522005],
#        [ 0.69842852,  0.06614192,  0.1260637 ],
#        [ 0.71968402,  0.63972258,  0.26906391],
#        [ 0.10744048,  0.60514499,  0.77606712],
#        [ 0.32825748,  0.7426097 ,  0.61835191],
#        [ 0.20366422,  0.36068633,  0.22225229],
#        [ 0.15571064,  0.78869831,  0.41425336],
#        [ 0.99644117,  0.3218098 ,  0.03129907],
#        [ 0.26777793,  0.30269442,  0.17798473],
#        [ 0.95856054,  0.76030378,  0.88535774],
#        [ 0.63334536,  0.00821585,  0.54328885],
#        [ 0.61875387,  0.90262691,  0.47184244]])

facecolor2 = numpy.array([[0.,0,0],
              [1.0,.0,0],
              [0.8,.0,0],
              [0.6,0,0],
              [0.4,0,0],

              [0,1,0],
              [0,0.8,0],
              [0,0.6,0],
              [0,0.4,0],

              [0,0,1],
              [0,0,0.8],
              [0,0,0.6],
              [0,0,0.4],

              [1,1,0],
              [1,0.8,0],
              [1.,0.6,0],
              [1,0.4,0],

              [1,0,1],
              [1.,0,0.8],
              [1,0,0.6],
              [1,0,0.4],
              [1,1,1],
              ]).reshape(22,3)

def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = numpy.mgrid[0:2*numpy.pi:20j, 0:numpy.pi:10j]
    x=numpy.cos(u)*numpy.sin(v)
    y=numpy.sin(u)*numpy.sin(v)
    z=numpy.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)
def skeleton2sphere_old(Skeleton,numInSphere,palmBaseTopScaleRatio,Hand_Sphere_Raidus):

    newSphere = numpy.empty((21,numInSphere,5))

    fingerIdxList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    ##thumb joints interpolate

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
    scaleMat = numpy.eye(3)*palmBaseTopScaleRatio
    tmpSke = Skeleton-Skeleton[9].reshape(1,3)
    basePalmJoints = numpy.dot(tmpSke[idx],scaleMat)+tmpSke[0].reshape(1,3)+Skeleton[9].reshape(1,3)

    for j in range(numInSphere):
        newSphere[0,j,1:4]= (basePalmJoints[-1]-basePalmJoints[0])*(j+1)/numInSphere+basePalmJoints[0]
        newSphere[0,j,4]=Hand_Sphere_Raidus[WRIST]



    ##palm joints except thumb joint interpolate
    for i, idx in enumerate(PALM[1:]):
        for j in range(numInSphere):
            newSphere[idx,j,1:4]= (Skeleton[idx]-basePalmJoints[i])*(j+1)/numInSphere+basePalmJoints[i]
            newSphere[idx,j,4]=(Hand_Sphere_Raidus[idx,]-Hand_Sphere_Raidus[WRIST,])*(j+1)/numInSphere+Hand_Sphere_Raidus[WRIST,]
    return newSphere
def skeleton2sphere_21jnt(Skeleton,numInSphere,palmBaseTopScaleRatio,Hand_Sphere_Raidus):

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

def skeleton2sphere_22jnt(Skeleton,numInSphere,palmBaseTopScaleRatio,Hand_Sphere_Raidus):

    newSphere = numpy.empty((22,numInSphere,5))


    fingerIdxList = [THUMB,INDEX,MIDDLE,RING,PINKY]
    ##thumb joints interpolate
    for i in range(22):
        newSphere[i,:,0]=i

    numThumbSphere=numInSphere+2
    j=1
    newSphere[0,-1,1:4]= (Skeleton[1]-Skeleton[WRIST])*(j+1)/numThumbSphere+Skeleton[WRIST]
    newSphere[0,-1,4]=Hand_Sphere_Raidus[WRIST]
    j=0
    newSphere[21,-1,1:4]= (Skeleton[1]-Skeleton[WRIST])*(j+1)/numThumbSphere+Skeleton[WRIST]
    newSphere[21,-1,4]=Hand_Sphere_Raidus[WRIST]

    for j in range(2,numThumbSphere,1):
        newSphere[1,j-2,1:4]= (Skeleton[1]-Skeleton[WRIST])*(j+1)/numThumbSphere+Skeleton[WRIST]
        newSphere[1,j-2,4]=(Hand_Sphere_Raidus[1,]-Hand_Sphere_Raidus[WRIST,])*(j+1)/numThumbSphere+Hand_Sphere_Raidus[WRIST,]


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
    numElbowSphere=numInSphere-2
    newSphere[21,0,1:4]=basePalmJoints[0]
    newSphere[21,0,4]=Hand_Sphere_Raidus[WRIST]
    for j in range(numElbowSphere):
        newSphere[21,j+1,1:4]= (basePalmJoints[-1]-basePalmJoints[0])*(j+1)/numElbowSphere+basePalmJoints[0]
        newSphere[21,j+1,4]=Hand_Sphere_Raidus[WRIST]

    # for j in range(numInSphere):
    #     newSphere[0,j,1:4]= (basePalmJoints[-1]-basePalmJoints[0])*(j+1)/numInSphere+basePalmJoints[0]
    #     newSphere[0,j,4]=Hand_Sphere_Raidus[WRIST]

    ##palm joints except thumb joint interpolate
    tmpwrist=[]
    numPalmSphere=numInSphere+1
    for i, idx in enumerate(PALM[1:]):
        j=0
        tmpwrist.append((Skeleton[idx]-basePalmJoints[i])*(j+1)/numPalmSphere+basePalmJoints[i])
        for j in range(1,numInSphere+1,1):
            newSphere[idx,j-1,1:4]= (Skeleton[idx]-basePalmJoints[i])*(j+1)/numPalmSphere+basePalmJoints[i]
            newSphere[idx,j-1,4]=(Hand_Sphere_Raidus[idx,]-Hand_Sphere_Raidus[WRIST,])*(j+1)/numPalmSphere+Hand_Sphere_Raidus[WRIST,]

    numWristSphere=numInSphere-2
    newSphere[0,0,1:4]=tmpwrist[0]
    newSphere[0,0,4]=Hand_Sphere_Raidus[WRIST]

    for j in range(numWristSphere):
        newSphere[0,j+1,1:4]= (tmpwrist[-1]-tmpwrist[0])*(j+1)/numWristSphere+tmpwrist[0]
        newSphere[0,j+1,4]=Hand_Sphere_Raidus[WRIST]


    return newSphere


def ShowDepthSphereMaskHist(setname,depth,numPoint,jnt_uvd,hand_points,Sphere):

    numInSphere=Sphere.shape[1]
    facecolor = numpy.zeros((numInSphere,3))
    facecolor[:,0]=numpy.arange(int(255.0/numInSphere)*numInSphere,0,-int(255.0/numInSphere))/255.0

    facecolor1 = numpy.zeros((numInSphere,3))
    facecolor1[:,1]=numpy.arange(int(255.0/numInSphere)*numInSphere,0,-int(255.0/numInSphere))/255.0

    facecolor2 = numpy.zeros((numInSphere,3))
    facecolor2[:,2]=numpy.arange(int(255.0/numInSphere)*numInSphere,0,-int(255.0/numInSphere))/255.0
    #Visualize the hand and the joints in 3D
    uvd = xyz_uvd.convert_depth_to_uvd(depth)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    axis_bounds = numpy.array([numpy.min(hand_points[:, 0]), numpy.max(hand_points[:, 0]),
                               numpy.min(hand_points[:, 1]), numpy.max(hand_points[:, 1]),
                               numpy.min(hand_points[:, 2]), numpy.max(hand_points[:, 2])])
    margin=100
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= margin
    axis_bounds[~mask] += margin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points',points0.shape

    margin=10
    roiDepth = depth[(int(numpy.min(jnt_uvd[:, 1]))-margin): (int(numpy.max(jnt_uvd[:, 1])+margin)),
               (int(numpy.min(jnt_uvd[:, 0]))-margin): (int(numpy.max(jnt_uvd[:, 0]))+margin)]
    upper = numpy.max(jnt_uvd[:,2])+100
    low = numpy.min(jnt_uvd[:,2])-100
    loc = numpy.where(numpy.logical_and(roiDepth<upper,roiDepth>low))
    normDepth = numpy.ones_like(roiDepth)*1.0
    normDepth[loc]=(roiDepth[loc]-low)/(upper-low)
    fig = plt.figure(figsize=(18,6))
    ax= fig.add_subplot(131)
    ax.imshow(normDepth,'gray')
    # ax.scatter(jnt_uvd[:,0],jnt_uvd[:,1])


    # fig = plt.figure(figsize=(18,6))
    # ax= fig.add_subplot(131)
    # ax.imshow(depth,'gray')
    # ax.scatter(jnt_uvd[:,0],jnt_uvd[:,1])

    ax = fig.add_subplot(132, projection='3d')
    # ax.scatter3D(points0[:, 0], points0[:, 1], points0[:, 2], s=1, marker='.',c='m')


    idx = [0]
    for i in range(4):
        for (xi,yi,zi,ri) in zip(Sphere[idx,i,1],Sphere[idx,i,2],Sphere[idx,i,3],Sphere[idx,i,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor1[i],alpha=1)
    # ax.scatter3D(Sphere[idx,i,1],Sphere[idx,i,2],Sphere[idx,i,3], s=100, marker='*',c='g')

    idx = [1]
    for i in range(numInSphere):
        for (xi,yi,zi,ri) in zip(Sphere[idx,i,1],Sphere[idx,i,2],Sphere[idx,i,3],Sphere[idx,i,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor2[i],alpha=1)

    idx = PALM[1:]
    for i in range(Sphere.shape[1]):
        for (xi,yi,zi,ri) in zip(Sphere[idx,i,1],Sphere[idx,i,2],Sphere[idx,i,3],Sphere[idx,i,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor[i],alpha=1)

    idx = MCP+DIP+TIP
    for i in range(Sphere.shape[1]):
        for (xi,yi,zi,ri) in zip(Sphere[idx,i,1],Sphere[idx,i,2],Sphere[idx,i,3],Sphere[idx,i,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor[i],alpha=1)


    # ax.scatter3D(Sphere[idx,:,1],Sphere[idx,i,2],Sphere[idx,i,3], s=100, marker='*',c='g')



    num_bar=21
    x_1 = numpy.arange(0,num_bar,1)
    ax= fig.add_subplot(133)
    bar_2 = ax.bar(x_1, numPoint,color ='b',width=1)
    plt.xticks(range(0,num_bar,1))
    # plt.ylabel('mean error (mm)',fontsize=20)
    plt.grid(b='on',which='both')
    plt.show()


def getPartLabel(setname,DepthImg,Sphere,pointCloudMargin):
    sphere_xyz = Sphere[:,1:4]
    numInSphere=Sphere.shape[0]/21

    #Visualize the hand and the joints in 3D
    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)


    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    Dists = cdist(Locates,sphere_xyz)
    # partLable = numpy.zeros((SubPixelNum))
    partLable  = numpy.floor(numpy.argmin(Dists,axis=1)/numInSphere)+1

    uvd_point = xyz_uvd.xyz2uvd(setname='nyu',xyz=points0)
    uvd_point[:,2]=partLable
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    partMap = numpy.ones(DepthImg.shape,dtype='uint32')

    partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]
    # plt.imshow(partMap)
    # plt.show()

    return partMap
def getMaskLabel_from_part_icvl(setname,DepthImg,Sphere,focal_len,numPointThresh,pointCloudMargin):

    #Visualize the hand and the joints in 3D
    numInSphere=Sphere.shape[0]/21
    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    sphere_xyz = Sphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    Dists = cdist(Locates,sphere_xyz)
    sphereLable = numpy.argmin(Dists,axis=1)

    """get part label  backgroud 0, hand parts start from 1 """
    partLable  = numpy.floor(sphereLable/numInSphere)
    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point[:,2]=partLable+1
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    partMap = numpy.zeros(DepthImg.shape,dtype='uint32')
    partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    """get visible mask """
    # partLable = numpy.zeros((SubPixelNum))
    numPoint=numpy.zeros((21,))
    numWristPalmPoint = numpy.zeros((4,))
    for ei, i in enumerate([5,9,13,17]):
        numWristPalmPoint[ei]+=numpy.where(sphereLable==i*numInSphere)[0].shape[0]
    numPoint[0]=numpy.sum(numWristPalmPoint)

    # print numPoint[0]

    """the visible of palm joint is the sum minus the num pixels assigned to wrist"""
    for i in range(1,21,1):
        numPoint[i]=numpy.where(partLable==i)[0].shape[0]
    # print numPoint[PALM[1:]]
    numPoint[PALM[1:]]-=numWristPalmPoint
    # print numPoint[PALM[1:]]

    numPoint = numPoint*numpy.mean(sphere_xyz[:,2])/focal_len
    # print 'dist',numpy.mean(jnt_xyz[:,2])
    # print numPoint[WRIST]
    # print numPoint[PALM]
    # print numPoint[MCP]
    # print numPoint[DIP]
    # print numPoint[TIP]
    boneMask = numpy.ones_like(numPoint)
    loc = numpy.where(numPoint<numPointThresh)
    boneMask[loc]=0
    # print 'num of visible', numpy.sum(boneMask)


    # plt.imshow(partMap)
    # plt.show()


    return boneMask,numPoint,partMap

def getMaskLabel_from_part(setname,DepthImg,Sphere,focal_len,numPointThresh,pointCloudMargin):

    #Visualize the hand and the joints in 3D
    numInSphere=Sphere.shape[0]/21
    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    sphere_xyz = Sphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    Dists = cdist(Locates,sphere_xyz)
    sphereLable = numpy.argmin(Dists,axis=1)

    """get part label  backgroud 0, hand parts start from 1 """
    partLable  = numpy.floor(sphereLable/numInSphere)
    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point[:,2]=partLable+1
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    partMap = numpy.zeros(DepthImg.shape,dtype='uint32')
    partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    """get visible mask """
    # partLable = numpy.zeros((SubPixelNum))
    numPoint=numpy.zeros((21,))
    numWristPalmPoint = numpy.zeros((4,))
    for ei, i in enumerate([5,9,13,17]):
        numWristPalmPoint[ei]+=numpy.where(sphereLable==i*numInSphere)[0].shape[0]
        # numWristPalmPoint[ei]+=numpy.where(sphereLable==(i*numInSphere+1))[0].shape[0]
        # numWristPalmPoint[ei]+=numpy.where(sphereLable==(i*numInSphere+2))[0].shape[0]
    numPoint[0]=numpy.sum(numWristPalmPoint)

    # print numPoint[0]

    """the visible of palm joint is the sum minus the num pixels assigned to wrist"""
    for i in range(1,21,1):
        numPoint[i]=numpy.where(partLable==i)[0].shape[0]
    # print numPoint[PALM[1:]]
    numPoint[PALM[1:]]-=numWristPalmPoint
    # print numPoint[PALM[1:]]

    numPoint = numPoint*numpy.mean(sphere_xyz[:,2])/focal_len
    # print 'dist',numpy.mean(jnt_xyz[:,2])
    # print numPoint[WRIST]
    # print numPoint[PALM]
    # print numPoint[MCP]
    # print numPoint[DIP]
    # print numPoint[TIP]
    boneMask = numpy.ones_like(numPoint)
    loc = numpy.where(numPoint<numPointThresh)
    boneMask[loc]=0
    # print 'num of visible', numpy.sum(boneMask)


    # plt.imshow(partMap)
    # plt.show()


    return boneMask,numPoint,partMap



def getMaskLabel_from_part_mega(setname,DepthImg,Sphere,focal_len,numPointThresh,pointCloudMargin):

    #Visualize the hand and the joints in 3D
    numInSphere=Sphere.shape[0]/21
    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    sphere_xyz = Sphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    Dists = cdist(Locates,sphere_xyz)
    sphereLable = numpy.argmin(Dists,axis=1)

    """get part label  backgroud 0, hand parts start from 1 """
    partLable  = numpy.floor(sphereLable/numInSphere)
    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point[:,2]=partLable+1
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    partMap = numpy.zeros(DepthImg.shape,dtype='uint32')
    partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    """get visible mask """
    # partLable = numpy.zeros((SubPixelNum))
    numPoint=numpy.zeros((21,))
    numWristPalmPoint = numpy.zeros((4,))
    for ei, i in enumerate([5,9,13,17]):
        numWristPalmPoint[ei]+=numpy.where(sphereLable==i*numInSphere)[0].shape[0]
        # numWristPalmPoint[ei]+=numpy.where(sphereLable==(i*numInSphere+1))[0].shape[0]
        # numWristPalmPoint[ei]+=numpy.where(sphereLable==(i*numInSphere+2))[0].shape[0]
    numPoint[0]=numpy.sum(numWristPalmPoint)

    # print numPoint[0]

    """the visible of palm joint is the sum minus the num pixels assigned to wrist"""
    for i in range(1,21,1):
        numPoint[i]=numpy.where(partLable==i)[0].shape[0]
    # print numPoint[PALM[1:]]
    numPoint[PALM[1:]]-=numWristPalmPoint
    # print numPoint[PALM[1:]]

    numPoint = numPoint*numpy.mean(sphere_xyz[:,2])/focal_len
    # print 'dist',numpy.mean(jnt_xyz[:,2])
    # print numPoint[WRIST]
    # print numPoint[PALM]
    # print numPoint[MCP]
    # print numPoint[DIP]
    # print numPoint[TIP]
    boneMask = numpy.ones_like(numPoint)
    loc = numpy.where(numPoint<numPointThresh)
    boneMask[loc]=0
    # print 'num of visible', numpy.sum(boneMask)


    # plt.imshow(partMap)
    # plt.show()


    return boneMask,numPoint,partMap


def getMaskLabel_sphere(setname,DepthImg,Sphere,focal_len,numPointThresh,pointCloudMargin):

    #Visualize the hand and the joints in 3D
    numInSphere=Sphere.shape[0]/21
    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    sphere_xyz = Sphere[:,1:4]
    sphere_radius = Sphere[:,4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    Dists = cdist(sphere_xyz,Locates)
    Dists.shape=(21,numInSphere,SubPixelNum)

    # tmp = numpy.mean(numpy.min(Dists,axis=-1),axis=-1)
    # print tmp[WRIST]
    # print tmp[PALM]
    # print tmp[MCP]
    # print tmp[DIP]
    # print tmp[TIP]


    count = numpy.zeros_like(Dists)
    loc =  numpy.where(Dists<sphere_radius.reshape(21,numInSphere,1))

    count[loc]=1

    numPoint = numpy.sum(numpy.sum(count,axis=-1),axis=-1)

    numPoint = numPoint*numpy.mean(sphere_xyz[:,2])/focal_len
    # print 'dist',numpy.mean(jnt_xyz[:,2])
    # print numPoint[WRIST]
    # print numPoint[PALM]
    # print numPoint[MCP]
    # print numPoint[DIP]
    # print numPoint[TIP]
    boneMask = numpy.ones_like(numPoint)
    loc = numpy.where(numPoint<numPointThresh)
    boneMask[loc]=0
    # print numpy.sum(boneMask)
    # print '*******'
    return boneMask,numPoint
#
#
#
#
# def getMaskLabel_old(setname,DepthImg,jnt_xyz,hand_points,radius):
#
#     #Visualize the hand and the joints in 3D
#     uvd = convert_depth_to_uvd(DepthImg)
#     xyz = uvd2xyz(setname=setname,uvd=uvd)
#     points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)
#
#
#     axis_bounds = numpy.array([numpy.min(jnt_xyz[:, 0]), numpy.max(jnt_xyz[:, 0]),
#                                numpy.min(jnt_xyz[:, 1]), numpy.max(jnt_xyz[:, 1]),
#                                numpy.min(jnt_xyz[:, 2]), numpy.max(jnt_xyz[:, 2])])
#     mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
#     axis_bounds[mask] -= pointCloudMargin
#     axis_bounds[~mask] += pointCloudMargin
#     mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
#     mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
#     mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
#     inumpyuts = mask1 & mask2 & mask3
#     points0 = points[inumpyuts]
#     # print 'num points', points0.shape
#
#     Gx = points0[:,0]
#     Gy = points0[:,1]
#     Gz = points0[:,2]
#
#     SubPixelNum=Gx.shape[0]
#     SubPixelInd=numpy.arange(0,SubPixelNum,1)
#     Locates = numpy.empty((SubPixelNum,3))
#
#     Locates[:,0]=Gx[SubPixelInd]
#     Locates[:,1]=Gy[SubPixelInd]
#     Locates[:,2]=Gz[SubPixelInd]
#
#     Dists = cdist(hand_points,Locates)
#     Dists.shape=(21,numInSphere,SubPixelNum)
#
#     # tmp = numpy.mean(numpy.min(Dists,axis=-1),axis=-1)
#     # print tmp[WRIST]
#     # print tmp[PALM]
#     # print tmp[MCP]
#     # print tmp[DIP]
#     # print tmp[TIP]
#
#
#     count = numpy.zeros_like(Dists)
#     loc =  numpy.where(Dists<radius.reshape(21,numInSphere,1))
#
#     count[loc]=1
#
#     num1 = numpy.sum(numpy.sum(count[0,0:4],axis=-1),axis=-1)
#     num2 = numpy.sum(numpy.sum(count[1,0:numInSphereThumb1],axis=-1),axis=-1)
#     # num3 =  numpy.sum(numpy.sum(count[PALM[1:],int(numInSphere/2):],axis=-1),axis=-1)
#     num3 =  numpy.sum(numpy.sum(count[PALM[1:],:],axis=-1),axis=-1)
#     num4= numpy.sum(numpy.sum(count[MCP+DIP+TIP,:],axis=-1),axis=-1)
#
#     numPoint = numpy.empty((21,),dtype='float32')
#     numPoint[0]=num1
#     numPoint[1]=num2
#     numPoint[PALM[1:]]=num3
#     numPoint[MCP+DIP+TIP]=num4
#
#     numPoint = numPoint*numpy.mean(jnt_xyz[:,2])/focal_len
#     # print 'dist',numpy.mean(jnt_xyz[:,2])
#     # print numPoint[WRIST]
#     # print numPoint[PALM]
#     # print numPoint[MCP]
#     # print numPoint[DIP]
#     # print numPoint[TIP]
#     boneMask = numpy.ones_like(numPoint)
#     loc = numpy.where(numPoint<500)
#     boneMask[loc]=0
#     # print numpy.sum(boneMask)
#     # print '*******'
#     return boneMask,numPoint



def getMask_mega(setname,DepthImg,Sphere,pointCloudMargin):
    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    sphere_xyz = Sphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    Dists = cdist(sphere_xyz,Locates)
    count = numpy.zeros_like(Dists)
    radius=Sphere[:,4]
    loc = numpy.where(Dists<radius.reshape(radius.shape[0],1))

    count[loc]=1
    numPoint = numpy.sum(count,axis=-1)*1.0/points0.shape[0]

    return numPoint

def ShowDepthSphereMaskHist2(title,setname,depth,numPoint,jnt_uvd,hand_points,Sphere,partMap):

    numInSphere=Sphere.shape[1]


    #Visualize the hand and the joints in 3D
    uvd = xyz_uvd.convert_depth_to_uvd(depth)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    axis_bounds = numpy.array([numpy.min(hand_points[:, 0]), numpy.max(hand_points[:, 0]),
                               numpy.min(hand_points[:, 1]), numpy.max(hand_points[:, 1]),
                               numpy.min(hand_points[:, 2]), numpy.max(hand_points[:, 2])])
    margin=30
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= margin
    axis_bounds[~mask] += margin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points',points0.shape
    mean_points=numpy.mean(points,axis=0)
    mean_sphere=numpy.mean(numpy.mean(Sphere[:,:,1:4],axis=0),axis=0)

    dist = numpy.sqrt(numpy.sum((mean_points-mean_sphere)**2,axis=-1))
    print('dist between joint center, sphere center', dist)

    margin=10
    roiDepth = depth[(int(numpy.min(jnt_uvd[:, 1]))-margin): (int(numpy.max(jnt_uvd[:, 1])+margin)),
               (int(numpy.min(jnt_uvd[:, 0]))-margin): (int(numpy.max(jnt_uvd[:, 0]))+margin)]
    upper = numpy.max(jnt_uvd[:,2])+100
    low = numpy.min(jnt_uvd[:,2])-100
    loc = numpy.where(numpy.logical_and(roiDepth<upper,roiDepth>low))
    normDepth = numpy.ones_like(roiDepth)*1.0
    normDepth[loc]=(roiDepth[loc]-low)/(upper-low)
    fig = plt.figure(figsize=(10,10))
    ax= fig.add_subplot(222)
    ax.imshow(normDepth,'gray')
    # ax.scatter(jnt_uvd[:,0],jnt_uvd[:,1])
    # num_bar=21
    # x_1 = numpy.arange(0,num_bar,1)
    # ax= fig.add_subplot(223)
    # bar_2 = ax.bar(x_1, numPoint,color ='b',width=1)
    # plt.xticks(range(0,num_bar,1))
    # # plt.ylabel('mean error (mm)',fontsize=20)
    # plt.grid(b='on',which='both')

    ax= fig.add_subplot(221)
    ax.imshow(depth,'gray')
    ax.scatter(jnt_uvd[:,0],jnt_uvd[:,1])

    ax= fig.add_subplot(224)
    ax.imshow(partMap,'jet')

    fig=plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    tmpidx = numpy.random.randint(0,points0.shape[0],int(points0.shape[0]/8.0))
    ax.scatter3D(points0[tmpidx, 0], points0[tmpidx, 1], points0[tmpidx, 2], s=20, marker='.',c='m')

    for idx in range(21):
        for (xi,yi,zi,ri) in zip(Sphere[idx,:,1],Sphere[idx,:,2],Sphere[idx,:,3],Sphere[idx,:,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor2[idx],alpha=1)
    plt.grid(b='on',which='both')
    plt.show()
    # ax.scatter3D(Sphere[idx,:,1],Sphere[idx,i,2],Sphere[idx,i,3], s=100, marker='*',c='g')




def ShowDepthSphere(title,setname,depth,Sphere):
    # uvd = xyz_uvd.convert_depth_to_uvd(depth)
    # xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    # points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    # ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], s=1, marker='.',c='m')

    for idx in range(21):
        for (xi,yi,zi,ri) in zip(Sphere[idx,:,1],Sphere[idx,:,2],Sphere[idx,:,3],Sphere[idx,:,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor2[idx],alpha=1)
    # for idx in [21]:
    #     for (xi,yi,zi,ri) in zip(Sphere[idx,:,1],Sphere[idx,:,2],Sphere[idx,:,3],Sphere[idx,:,4]):
    #         (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
    #         ax.plot_wireframe(xs, ys, zs,color=facecolor2[idx],alpha=1)
    plt.grid(b='on',which='both')
    plt.show()
    # ax.scatter3D(Sphere[idx,:,1],Sphere[idx,i,2],Sphere[idx,i,3], s=100, marker='*',c='g')

def getPart_mega(setname,DepthImg,Sphere,numInSphere,numJnt,pointCloudMargin):

    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    sphere_xyz = Sphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    Dists = cdist(Locates,sphere_xyz,metric='euclidean')

    sphereLable = numpy.argmin(Dists,axis=1)
    loc=(range(Dists.shape[0]),sphereLable)
    tmploc = numpy.where(Dists[loc]>50)

    sphereLable+=numInSphere
    sphereLable[tmploc]=0
    """get part label  backgroud 0, hand parts start from 1 """
    partLable  = numpy.floor(sphereLable/numInSphere)

    # loc = numpy.where(partLable<=2)
    # partLable[loc]=0
    # # Dists.shape=(Dists.shape[0],numJnt,numInSphere)
    # tmpDist=Dists[:,0:numInSphere*2]
    # loc2 = numpy.where(tmpDist[loc]<100)
    # partLable[loc][loc2[0]]=2


    partLable[numpy.where(partLable==1)]=2
    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point[:,2]=partLable

    # loc = numpy.where(sphereLable==)
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    partMap = numpy.zeros(DepthImg.shape,dtype='uint32')

    pointColor=numpy.ones((uvd_point.shape[0],numJnt+1,3))*facecolor2.reshape(1,numJnt+1,3)
    colorlabel=pointColor[(range(0,uvd_point.shape[0],1),uvd_point[:,2])]

    ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='float32')
    ColorMap[uvd_point[:,1],uvd_point[:,0],:]=colorlabel
    partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    """get visible mask """
    numPoint=numpy.zeros((21,))
    """the visible of palm joint is the sum minus the num pixels assigned to wrist"""
    for i in range(2,22,1):
        numPoint[i-1]=numpy.where(partLable==i)[0].shape[0]



    return partMap,numPoint*1.0/points0.shape[0],ColorMap





def getPart_mega2(setname,DepthImg,Sphere,numInSphere,numJnt,pointCloudMargin):

    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)

    sphere_xyz = Sphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    Dists = cdist(Locates,sphere_xyz,metric='euclidean')
    maxD=numpy.max(Dists)

    # expPointDepth=numpy.ones((SubPixelNum,sphere_xyz.shape[0]))*Locates[:,2].reshape(SubPixelNum,1)
    # expSphZ=numpy.ones((SubPixelNum,sphere_xyz.shape[0]))*sphere_xyz[:,2].reshape(1,sphere_xyz.shape[0])
    # inDepthLoc = numpy.where(expPointDepth<expSphZ)
    # Dists[inDepthLoc]=10000000
    #
    for s in range(sphere_xyz.shape[0]):
        sphereBehindPoint = numpy.where(Locates[:,2]+15<sphere_xyz[s,2])
        sphereFrontPoint = numpy.where(Locates[:,2]+15>sphere_xyz[s,2])
        # print(inDepthLoc[0].shape)
        Dists[sphereFrontPoint,s]=maxD+1

        # if s>(1+4*numInSphere*4+3*numInSphere):
        #     print(s,s/numInSphere)
        #     fig=plt.figure(figsize=(10,10))
        #     ax = fig.add_subplot(111, projection='3d')
        #     tmpPoint=Locates[sphereFrontPoint]
        #     tmp2=Locates[sphereBehindPoint]
        #     ax.scatter3D(tmpPoint[:, 0], tmpPoint[:, 1], tmpPoint[:, 2], s=1, marker='.',c='m')
        #     ax.scatter3D(tmp2[:, 0], tmp2[:, 1], tmp2[:, 2], s=1, marker='.',c='g')
        #     ax.scatter3D(sphere_xyz[:, 0], sphere_xyz[:, 1], sphere_xyz[:, 2], s=100, marker='.',c='y')
        #     ax.scatter3D(sphere_xyz[s, 0], sphere_xyz[s, 1], sphere_xyz[s, 2], s=1000, marker='.',c='r')
        #     plt.show()
    sphere_radius = Sphere[:,4]
    tmploc = numpy.where(Dists>sphere_radius.reshape(1,numInSphere*numJnt)*2+10)
    Dists[tmploc]=maxD

    sphereLable = numpy.argmin(Dists,axis=1)
    loc=(range(Dists.shape[0]),sphereLable)
    tmploc = numpy.where(Dists[loc]>50)

    sphereLable+=numInSphere
    sphereLable[tmploc]=0

    """get part label  backgroud 0, hand parts start from 1 """
    partLable  = numpy.floor(sphereLable/numInSphere)

    # loc = numpy.where(partLable<=2)
    # partLable[loc]=0
    # # Dists.shape=(Dists.shape[0],numJnt,numInSphere)
    # tmpDist=Dists[:,0:numInSphere*2]
    # loc2 = numpy.where(tmpDist[loc]<100)
    # partLable[loc][loc2[0]]=2


    partLable[numpy.where(partLable==1)]=2
    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point[:,2]=partLable

    # loc = numpy.where(sphereLable==)
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    partMap = numpy.zeros(DepthImg.shape,dtype='uint32')

    pointColor=numpy.ones((uvd_point.shape[0],numJnt+1,3))*facecolor2.reshape(1,numJnt+1,3)
    colorlabel=pointColor[(range(0,uvd_point.shape[0],1),uvd_point[:,2])]

    ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='float32')
    ColorMap[uvd_point[:,1],uvd_point[:,0],:]=colorlabel
    partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    """get visible mask """
    numPoint=numpy.zeros((21,))
    """the visible of palm joint is the sum minus the num pixels assigned to wrist"""
    for i in range(2,22,1):
        numPoint[i-1]=numpy.where(partLable==i)[0].shape[0]



    return partMap,numPoint*1.0/points0.shape[0],ColorMap


def sort_sphere(Sphere,numJnt,numInSphere):
    fingIdx=[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]
    palmIdx=[0,1,5,9,13,17]
    tmpSphere=Sphere.reshape(numJnt,numInSphere,5)

    newSphere=numpy.empty_like(tmpSphere)

    palmSphere =tmpSphere[palmIdx]
    idxsort = numpy.argsort(-numpy.mean(palmSphere[:,:,3],axis=1))
    newSphere[:len(palmIdx)]=palmSphere[idxsort]

    fingSphere = tmpSphere[fingIdx]
    idxsort = numpy.argsort(-numpy.mean(fingSphere[:,:,3],axis=1))
    newSphere[len(palmIdx):]=fingSphere[idxsort]
    print(numpy.mean(newSphere[:,:,3],axis=1),newSphere[:,0,0])
    newSphere.shape=(numJnt*numInSphere,5)


    return newSphere
def sort_sphere1_22jnt(Sphere,numJnt,numInSphere):
    fingIdx=[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]
    palmIdx=[0,1,5,9,13,17,21]
    tmpSphere=Sphere.reshape(numJnt,numInSphere,5)

    newSphere=numpy.empty_like(tmpSphere)

    fingSphere = tmpSphere[fingIdx]
    idxsort = numpy.argsort(-numpy.mean(fingSphere[:,:,3],axis=1))
    newSphere[:len(fingIdx)]=fingSphere[idxsort]

    palmSphere =tmpSphere[palmIdx]
    idxsort = numpy.argsort(-numpy.mean(palmSphere[:,:,3],axis=1))
    newSphere[len(fingIdx):]=palmSphere[idxsort]


    print(numpy.mean(newSphere[:,:,3],axis=1),newSphere[:,0,0])
    newSphere.shape=(numJnt*numInSphere,5)


    return newSphere

def sort_sphere1_21jnt(Sphere,numJnt,numInSphere):
    fingIdx=[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]
    palmIdx=[0,1,5,9,13,17]
    tmpSphere=Sphere.reshape(numJnt,numInSphere,5)

    newSphere=numpy.empty_like(tmpSphere)

    fingSphere = tmpSphere[fingIdx]
    idxsort = numpy.argsort(-numpy.mean(fingSphere[:,:,3],axis=1))
    newSphere[:len(fingIdx)]=fingSphere[idxsort]

    palmSphere =tmpSphere[palmIdx]
    idxsort = numpy.argsort(-numpy.mean(palmSphere[:,:,3],axis=1))
    newSphere[len(fingIdx):]=palmSphere[idxsort]


    # print(numpy.mean(newSphere[:,:,3],axis=1),newSphere[:,0,0])
    newSphere.shape=(numJnt*numInSphere,5)


    return newSphere


def arange_sphere1_21jnt(Sphere,numJnt,numInSphere):
    tmpSphere=Sphere.reshape(numJnt,numInSphere,5)
    newIdx = [0,1,5,9,13,17,4,8,12,16,20,3,7,11,15,19,2,6,10,14,18]
    newSphere=tmpSphere[newIdx]
    newSphere.shape=(numJnt*numInSphere,5)
    return newSphere


def sort_sphere2(Sphere,numJnt,numInSphere):

    tmpSphere=Sphere.reshape(numJnt,numInSphere,5)
    idxsort = numpy.argsort(-numpy.mean(tmpSphere[:,:,3],axis=1))
    newSphere=tmpSphere[idxsort]
    print(numpy.mean(newSphere[:,:,3],axis=1),newSphere[:,0,0])
    newSphere.shape=(numJnt*numInSphere,5)


    return newSphere


def getPart_mega_21jnt_handobject(setname,DepthImg,Sphere,numInSphere,numJnt,pointCloudMargin):

    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)



    newSphere=sort_sphere1_21jnt(Sphere,numJnt,numInSphere)
    # newSphere=arange_sphere1_21jnt(Sphere,numJnt,numInSphere)

    sphere_xyz = newSphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    if points0.size == 0:
        uvd_point=[]
        partMap = numpy.ones(DepthImg.shape,dtype='uint32')*21
        ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='uint8')
        ColorMap[:,:]=[255,255,255]
        return uvd_point,partMap,ColorMap

    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    radius=newSphere[:,4]
    Dists = cdist(Locates,sphere_xyz,metric='euclidean')/radius.reshape(1,newSphere.shape[0])
    maxD=numpy.max(Dists)

    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    # xyz_point_label=numpy.empty((uvd_point.shape[0],4))
    # xyz_point_label[:,3]=22
    uvd_point[:,2]=21
    partMap = numpy.ones(DepthImg.shape,dtype='uint32')*21

    for s in range(sphere_xyz.shape[0]):
        sphereFrontPoint = numpy.where(Locates[:,2]+0>sphere_xyz[s,2])
        # loc = numpy.where(Dists[:,s]>newSphere[s,4]*2)
        # Dists[loc,s]=maxD+1
        Dists[sphereFrontPoint,s]=maxD*500

    # dist_p_s = cdist(Locates, Spheres[1:4, :].T)
    # closest_sphere = numpy.argmin(dist_p_s,axis=-1)
    # closest_dist = dist_p_s[:,closest_sphere]
    # tmp=numpy.ones_like(dist_p_s)*(Spheres[4,:].reshape(1,totalSphere))
    # closest_sphere_radius = tmp[:,closest_sphere]
    # Cost_D = numpy.abs(closest_dist-closest_sphere_radius)


    for s in range(sphere_xyz.shape[0]):
        # sphereFrontPoint = numpy.where(Locates[:,2]>sphere_xyz[s,2])
        # Dists[sphereFrontPoint,s]=maxD*500

        tmpLable = numpy.argmin(Dists,axis=1)

        mask = numpy.logical_and(Dists[:,s]<0.8,tmpLable==s)
        loc=numpy.where(mask==1)
        # loc=numpy.where(tmpLable==s)
        uvd_point[loc,2]=newSphere[s,0]
        partMap[uvd_point[loc,1],uvd_point[loc,0]]=uvd_point[loc,2]
        Dists[loc]=maxD*500
        # loc=numpy.where(tmpLable==s)
        # xyz_point_label[loc,3]=newSphere[s,0]
        # if (s+1)%numInSphere==0:
        #     plt.imshow(partMap,cmap='jet')
        #     plt.title('jnt%d'%newSphere[s,0])
        #     plt.show()


    # partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    # partMap[numpy.asarray(numpy.round(uvd_point[:,1]),dtype='uint32'),numpy.asarray(numpy.round(uvd_point[:,0]),dtype='uint32')]=uvd_point[:,2]

    pointColor=numpy.ones((uvd_point.shape[0],numJnt+1,3))*facecolor2.reshape(1,numJnt+1,3)
    colorlabel=pointColor[(range(0,uvd_point.shape[0],1),uvd_point[:,2])]

    ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='uint8')
    ColorMap[:,:]=[255,255,255]
    ColorMap[uvd_point[:,1],uvd_point[:,0]]=colorlabel*255
    # ColorMap[numpy.asarray(numpy.round(uvd_point[:,1]),dtype='uint32'),numpy.asarray(numpy.round(uvd_point[:,0]),dtype='uint32')]=colorlabel

    return uvd_point,partMap,ColorMap

def getPart_mega_21jnt_tmptest(setname,DepthImg,Sphere,numInSphere,numJnt,pointCloudMargin):

    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)



    newSphere=sort_sphere1_21jnt(Sphere,numJnt,numInSphere)
    # newSphere=arange_sphere1_21jnt(Sphere,numJnt,numInSphere)

    sphere_xyz = newSphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    if points0.size == 0:
        uvd_point=[]
        partMap = numpy.ones(DepthImg.shape,dtype='uint32')*21
        ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='uint8')
        ColorMap[:,:]=[255,255,255]
        return uvd_point,partMap,ColorMap

    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    radius=newSphere[:,4]
    Dists = cdist(Locates,sphere_xyz,metric='euclidean')/radius.reshape(1,newSphere.shape[0])
    maxD=numpy.max(Dists)

    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    # xyz_point_label=numpy.empty((uvd_point.shape[0],4))
    # xyz_point_label[:,3]=22
    uvd_point[:,2]=21
    partMap = numpy.ones(DepthImg.shape,dtype='uint32')*21

    for s in range(sphere_xyz.shape[0]):
        sphereFrontPoint = numpy.where(Locates[:,2]+0>sphere_xyz[s,2])
        # loc = numpy.where(Dists[:,s]>newSphere[s,4]*2)
        # Dists[loc,s]=maxD+1
        Dists[sphereFrontPoint,s]=maxD*500
    maxR=numpy.max(radius)
    for s in range(sphere_xyz.shape[0]):
        # sphereFrontPoint = numpy.where(Locates[:,2]>sphere_xyz[s,2])
        # Dists[sphereFrontPoint,s]=maxD*500

        tmpLable = numpy.argmin(Dists,axis=1)
        mask = numpy.logical_and(Dists[:,s]<2,tmpLable==s)
        loc=numpy.where(mask==1)
        # loc=numpy.where(tmpLable==s)
        uvd_point[loc,2]=newSphere[s,0]
        partMap[uvd_point[loc,1],uvd_point[loc,0]]=uvd_point[loc,2]
        Dists[loc]=maxD*500
        # loc=numpy.where(tmpLable==s)
        # xyz_point_label[loc,3]=newSphere[s,0]
        # if (s+1)%numInSphere==0:
        #     plt.imshow(partMap,cmap='jet')
        #     plt.title('jnt%d'%newSphere[s,0])
        #     plt.show()


    # partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    # partMap[numpy.asarray(numpy.round(uvd_point[:,1]),dtype='uint32'),numpy.asarray(numpy.round(uvd_point[:,0]),dtype='uint32')]=uvd_point[:,2]

    pointColor=numpy.ones((uvd_point.shape[0],numJnt+1,3))*facecolor2.reshape(1,numJnt+1,3)
    colorlabel=pointColor[(range(0,uvd_point.shape[0],1),uvd_point[:,2])]

    ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='uint8')
    ColorMap[:,:]=[255,255,255]
    ColorMap[uvd_point[:,1],uvd_point[:,0]]=colorlabel*255
    # ColorMap[numpy.asarray(numpy.round(uvd_point[:,1]),dtype='uint32'),numpy.asarray(numpy.round(uvd_point[:,0]),dtype='uint32')]=colorlabel

    return uvd_point,partMap,ColorMap

def getPart_mega_21jnt(setname,DepthImg,Sphere,numInSphere,numJnt,pointCloudMargin):

    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)



    # newSphere=sort_sphere1_21jnt(Sphere,numJnt,numInSphere)
    newSphere=arange_sphere1_21jnt(Sphere,numJnt,numInSphere)

    sphere_xyz = newSphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    if points0.size == 0:
        uvd_point=[]
        partMap = numpy.ones(DepthImg.shape,dtype='uint32')*21
        ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='uint8')
        ColorMap[:,:]=[255,255,255]
        return uvd_point,partMap,ColorMap

    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    radius=newSphere[:,4]
    Dists = cdist(Locates,sphere_xyz,metric='euclidean')/radius.reshape(1,newSphere.shape[0])
    maxD=numpy.max(Dists)

    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    # xyz_point_label=numpy.empty((uvd_point.shape[0],4))
    # xyz_point_label[:,3]=22
    uvd_point[:,2]=21
    partMap = numpy.ones(DepthImg.shape,dtype='uint32')*21

    for s in range(sphere_xyz.shape[0]):
        sphereFrontPoint = numpy.where(Locates[:,2]+10>sphere_xyz[s,2])
        # loc = numpy.where(Dists[:,s]>newSphere[s,4]*2)
        # Dists[loc,s]=maxD+1
        Dists[sphereFrontPoint,s]=maxD*500
    maxR=numpy.max(radius)
    for s in range(sphere_xyz.shape[0]):
        tmpLable = numpy.argmin(Dists,axis=1)
        mask = numpy.logical_and(Dists[:,s]<maxD*500,tmpLable==s)
        loc=numpy.where(mask==1)
        # loc=numpy.where(tmpLable==s)
        uvd_point[loc,2]=newSphere[s,0]
        partMap[uvd_point[loc,1],uvd_point[loc,0]]=uvd_point[loc,2]
        Dists[loc]=maxD*500
        # loc=numpy.where(tmpLable==s)
        # xyz_point_label[loc,3]=newSphere[s,0]
        # if (s+1)%numInSphere==0:
        #     plt.imshow(partMap,cmap='jet')
        #     plt.title('jnt%d'%newSphere[s,0])
        #     plt.show()


    # partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    # partMap[numpy.asarray(numpy.round(uvd_point[:,1]),dtype='uint32'),numpy.asarray(numpy.round(uvd_point[:,0]),dtype='uint32')]=uvd_point[:,2]

    pointColor=numpy.ones((uvd_point.shape[0],numJnt+1,3))*facecolor2.reshape(1,numJnt+1,3)
    colorlabel=pointColor[(range(0,uvd_point.shape[0],1),uvd_point[:,2])]

    ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='uint8')
    ColorMap[:,:]=[255,255,255]
    ColorMap[uvd_point[:,1],uvd_point[:,0]]=colorlabel*255
    # ColorMap[numpy.asarray(numpy.round(uvd_point[:,1]),dtype='uint32'),numpy.asarray(numpy.round(uvd_point[:,0]),dtype='uint32')]=colorlabel

    return uvd_point,partMap,ColorMap

def getPart_mega_22jnt(setname,DepthImg,Sphere,numInSphere,numJnt,pointCloudMargin):

    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)



    newSphere=sort_sphere1_22jnt(Sphere,numJnt,numInSphere)
    sphere_xyz = newSphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    radius=newSphere[:,4]
    Dists = cdist(Locates,sphere_xyz,metric='euclidean')/radius.reshape(1,newSphere.shape[0])
    maxD=numpy.max(Dists)

    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)

    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    # xyz_point_label=numpy.empty((uvd_point.shape[0],4))
    # xyz_point_label[:,3]=22
    uvd_point[:,2]=21
    partMap = numpy.ones(DepthImg.shape,dtype='uint32')*21

    for s in range(sphere_xyz.shape[0]):
        sphereFrontPoint = numpy.where(Locates[:,2]+10>sphere_xyz[s,2])
        # loc = numpy.where(Dists[:,s]>newSphere[s,4]*2)
        # Dists[loc,s]=maxD+1
        Dists[sphereFrontPoint,s]=maxD*500
    maxR=numpy.max(radius)
    for s in range(sphere_xyz.shape[0]):
        tmpLable = numpy.argmin(Dists,axis=1)
        mask = numpy.logical_and(Dists[:,s]<2*maxR,tmpLable==s)
        loc=numpy.where(mask==1)
        # loc=numpy.where(tmpLable==s)
        uvd_point[loc,2]=newSphere[s,0]
        partMap[uvd_point[loc,1],uvd_point[loc,0]]=uvd_point[loc,2]
        Dists[loc]=maxD*500
        # loc=numpy.where(tmpLable==s)
        # xyz_point_label[loc,3]=newSphere[s,0]
        # if (s+1)%numInSphere==0:
        #     plt.imshow(partMap,cmap='jet')
        #     plt.title('jnt%d'%newSphere[s,0])
        #     plt.show()


    # partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    # partMap[numpy.asarray(numpy.round(uvd_point[:,1]),dtype='uint32'),numpy.asarray(numpy.round(uvd_point[:,0]),dtype='uint32')]=uvd_point[:,2]

    pointColor=numpy.ones((uvd_point.shape[0],numJnt,3))*facecolor2.reshape(1,numJnt,3)
    colorlabel=pointColor[(range(0,uvd_point.shape[0],1),uvd_point[:,2])]

    ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='uint8')
    ColorMap[:,:]=[255,255,255]
    ColorMap[uvd_point[:,1],uvd_point[:,0]]=colorlabel*255
    # ColorMap[numpy.asarray(numpy.round(uvd_point[:,1]),dtype='uint32'),numpy.asarray(numpy.round(uvd_point[:,0]),dtype='uint32')]=colorlabel

    return uvd_point,partMap,ColorMap


def getPart_mega_ori(setname,DepthImg,Sphere,numInSphere,numJnt,pointCloudMargin):

    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)



    newSphere=Sphere.copy()
    sphere_xyz = newSphere[:,1:4]
    axis_bounds = numpy.array([numpy.min(sphere_xyz[:, 0]), numpy.max(sphere_xyz[:, 0]),
                               numpy.min(sphere_xyz[:, 1]), numpy.max(sphere_xyz[:, 1]),
                               numpy.min(sphere_xyz[:, 2]), numpy.max(sphere_xyz[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= pointCloudMargin
    axis_bounds[~mask] += pointCloudMargin
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]
    # print 'num points', points0.shape

    Gx = points0[:,0]
    Gy = points0[:,1]
    Gz = points0[:,2]

    SubPixelNum=Gx.shape[0]
    SubPixelInd=numpy.arange(0,SubPixelNum,1)
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]

    radius=newSphere[:,4]
    Dists = cdist(Locates,sphere_xyz,metric='euclidean')/radius.reshape(1,newSphere.shape[0])

    sphereLable = numpy.argmin(Dists,axis=1)
    # loc=(range(Dists.shape[0]),sphereLable)
    # tmploc = numpy.where(Dists[loc]>50)
    # sphereLable[tmploc]=0
    """get part label  backgroud 0, hand parts start from 1 """
    partLable  = numpy.floor(sphereLable/numInSphere)

    uvd_point = xyz_uvd.xyz2uvd(setname=setname,xyz=points0)
    uvd_point[:,2]=partLable

    # loc = numpy.where(sphereLable==)
    uvd_point=numpy.asarray(numpy.round(uvd_point),dtype='uint32')
    partMap = numpy.ones(DepthImg.shape,dtype='uint32')*21
    partMap[uvd_point[:,1],uvd_point[:,0]]=uvd_point[:,2]

    pointColor=numpy.ones((uvd_point.shape[0],numJnt+1,3))*facecolor2.reshape(1,numJnt+1,3)
    colorlabel=pointColor[(range(0,uvd_point.shape[0],1),uvd_point[:,2])]

    ColorMap = numpy.ones((DepthImg.shape[0],DepthImg.shape[1],3),dtype='uint8')
    ColorMap[:,:]=[255,255,255]
    ColorMap[uvd_point[:,1],uvd_point[:,0]]=colorlabel*255

    """get visible mask """
    # numPoint=numpy.zeros((21,))
    # """the visible of palm joint is the sum minus the num pixels assigned to wrist"""
    # for i in range(0,21,1):
    #     numPoint[i]=numpy.where(partLable==i)[0].shape[0]
    return  partMap,ColorMap

def ShowPointCloudFromDepth(setname,depth,hand_points,Sphere):

    #Visualize the hand and the joints in 3D
    uvd = xyz_uvd.convert_depth_to_uvd(depth)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    points = xyz.reshape(xyz.shape[0]*xyz.shape[1], 3)


    axis_bounds = numpy.array([numpy.min(hand_points[:, 0]), numpy.max(hand_points[:, 0]),
                               numpy.min(hand_points[:, 1]), numpy.max(hand_points[:, 1]),
                               numpy.min(hand_points[:, 2]), numpy.max(hand_points[:, 2])])
    mask = numpy.array([1, 0, 1, 0, 1, 0], dtype=bool)
    axis_bounds[mask] -= 20
    axis_bounds[~mask] += 20
    mask1 = (points[:, 0] >= axis_bounds[0]) & (points[:, 0] <= axis_bounds[1])
    mask2 = (points[:, 1] >= axis_bounds[2]) & (points[:, 1] <= axis_bounds[3])
    mask3 = (points[:, 2] >= axis_bounds[4]) & (points[:, 2] <= axis_bounds[5])
    inumpyuts = mask1 & mask2 & mask3
    points0 = points[inumpyuts]


    fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(121)
    # ax.imshow(depth,'gray')

    ax = fig.add_subplot(111, projection='3d')
    tmpidx = numpy.random.randint(0,points0.shape[0],int(points0.shape[0]/8.0))
    ax.scatter3D(points0[tmpidx, 0], points0[tmpidx, 1], points0[tmpidx, 2], s=10, marker='.')
    # ax.scatter3D(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], s=150, marker='*',c='g')
    # for (xi,yi,zi,ri) in zip(Sphere[:,1],Sphere[:,2],Sphere[:,3],Sphere[:,4],):
    #     (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
    #     ax.plot_wireframe(xs, ys, zs,rstride=4,cstride=4,color="r")

    for idx in range(21):
        for (xi,yi,zi,ri) in zip(Sphere[idx,:,1],Sphere[idx,:,2],Sphere[idx,:,3],Sphere[idx,:,4]):
            (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
            ax.plot_wireframe(xs, ys, zs,color=facecolor2[idx],alpha=1)
    plt.grid(b='on',which='both')
    plt.show()
    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    # ax.azim = -30
    # ax.elev = 10

    plt.show()


