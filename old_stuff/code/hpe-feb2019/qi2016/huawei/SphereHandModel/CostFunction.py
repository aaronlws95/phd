import numpy
from src.SphereHandModel.utils import xyz_uvd
from scipy.spatial.distance import cdist
# from src.SphereHandModel.ShowSamples import *




def cost_function(setname,DepthImg, inSpheres, Center, SilhouetteDistImg,SubPixelNum):
    # if setname=='mega':
    #     Center = [320,240]
    # if setname=='icvl':
    #     Center = [160,120]
    numShpere=inSpheres.shape[1]
    totalSphere = inSpheres.shape[0]*inSpheres.shape[1]
    Spheres=inSpheres.reshape(inSpheres.shape[0]*inSpheres.shape[1],5).T
    uvd = xyz_uvd.convert_depth_to_uvd(DepthImg)
    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
    # points = xyz.reshape(xyz.shape[0],xyz.shape[1], 3)
    Gx = xyz[:,:,0]
    Gy = xyz[:,:,1]
    Gz = xyz[:,:,2]

    PixelInd = numpy.where( DepthImg > 0 )
    # print 'num of hand points', PixelInd[0].shape[0]
    if PixelInd[0].shape[0]<SubPixelNum:
        SubPixelNum = PixelInd[0].shape[0]-10
    tmp = numpy.random.randint(0,PixelInd[0].shape[0],SubPixelNum)
    SubPixelInd =(PixelInd[0][tmp],PixelInd[1][tmp])


    # SubD = numpy.zeros(1, SubPixelNum)
    # DepthDiscrepancyThreshold = 10
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]
    # ShowPointCloud(points0=Locates,hand_points=Spheres[1:4,SLICE_IDX_SKE_FROM_SPHERE].T,Sphere=Spheres.T)
    Dists = numpy.abs(cdist(Locates, Spheres[1:4, :].T) -Spheres[4,:].reshape(1,totalSphere))
    # Dists = numpy.abs(cdist(Locates, Spheres[1:4, :].T) - numpy.ones((SubPixelNum, Spheres.shape[-1]))*Spheres[4,:].T)
    # SubD = Dists.T
    SubD = numpy.min(Dists.T,axis=0)
    # print SubD.shape

    B = numpy.zeros((1, totalSphere))
    sphereUVD = xyz_uvd.xyz2uvd(setname=setname,xyz=Spheres[1:4,:].T)
    u = numpy.asarray(numpy.round(sphereUVD[:,0]),dtype='int16')
    v =  numpy.asarray(numpy.round(sphereUVD[:,1]),dtype='int16')

    # % check whether u or v is out of the range
    if (max(u) >= Center[0]*2) or (min(u) <= 0) or (max(v) >= Center[1]*2) or (min(v) <= 0):
        B = 1000 * numpy.ones((1,totalSphere))
    else:
        DepthProj = DepthImg[(v,u)]
        DepthSphere = sphereUVD[:,2]
        # % Find the valid projected point
        ValidSpheresProjInd = numpy.where(DepthProj>0)
        InValidSpheresProjInd = numpy.where(0 == DepthProj)
        # templength = ValidSpheresProjInd[0].shape[0]
        temp1 = DepthProj[ValidSpheresProjInd] - DepthSphere[ValidSpheresProjInd]
        temp1[numpy.where(temp1<0)]=0
        # temp2 = numpy.max([numpy.zeros((templength,)), temp1],axis=0)
        # %B(ValidSpheresProjInd) = min([DepthDiscrepancyThreshold*ones(1,templength) temp2])
        B[:,ValidSpheresProjInd]= temp1
        invalidVU = (v[InValidSpheresProjInd],u[InValidSpheresProjInd])
        B[:,InValidSpheresProjInd] = SilhouetteDistImg[invalidVU]
        # print 'InValidSpheresProjInd',InValidSpheresProjInd, SilhouetteDistImg[invalidVU]
        # B[InValidSpheresProjInd] = SilhouetteDistImg(indices(InValidSpheresProjInd))
    L_Concise = numpy.zeros((7,3*numShpere,3*numShpere))
    THUMB_SHPERE=range(2*numShpere,5*numShpere,1)
    INDEX_SPHERE=range(6*numShpere,9*numShpere,1)
    MIDDLE_SPHERE=range(10*numShpere,13*numShpere,1)
    RING_SPHERE=range(14*numShpere,17*numShpere,1)
    PINKY_SPHERE=range(18*numShpere,21*numShpere,1)

    Thumb_Shperes = Spheres[:,THUMB_SHPERE]
    Index_Spheres = Spheres[:,INDEX_SPHERE]
    Middle_Spheres = Spheres[:,MIDDLE_SPHERE]
    Ring_Spheres = Spheres[:,RING_SPHERE]
    Small_Spheres = Spheres[:,PINKY_SPHERE]


    L_Concise[0] = cdist(Index_Spheres[-1,:].T.reshape(3*numShpere,1), - Middle_Spheres[-1,:].T.reshape(3*numShpere,1)) \
                               - cdist(Index_Spheres[1:4,:].T, Middle_Spheres[1:4,:].T)
    L_Concise[1] = cdist(Middle_Spheres[-1,:].T.reshape(3*numShpere,1), - Ring_Spheres[-1,:].T.reshape(3*numShpere,1)) \
                        - cdist(Middle_Spheres[1:4,:].T, Ring_Spheres[1:4,:].T)
    L_Concise[2] = cdist(Ring_Spheres[-1,:].T.reshape(3*numShpere,1), - Small_Spheres[-1,:].T.reshape(3*numShpere,1)) \
                         - cdist(Ring_Spheres[1:4,:].T, Small_Spheres[1:4,:].T)

    L_Concise[3] = cdist(Thumb_Shperes[-1,:].T.reshape(3*numShpere,1), - Middle_Spheres[-1,:].T.reshape(3*numShpere,1)) \
                               - cdist(Index_Spheres[1:4,:].T, Middle_Spheres[1:4,:].T)
    L_Concise[4] = cdist(Thumb_Shperes[-1,:].T.reshape(3*numShpere,1), - Ring_Spheres[-1,:].T.reshape(3*numShpere,1)) \
                        - cdist(Middle_Spheres[1:4,:].T, Ring_Spheres[1:4,:].T)
    L_Concise[5] = cdist(Thumb_Shperes[-1,:].T.reshape(3*numShpere,1), - Small_Spheres[-1,:].T.reshape(3*numShpere,1)) \
                         - cdist(Ring_Spheres[1:4,:].T, Small_Spheres[1:4,:].T)
    L_Concise[6] = cdist(Thumb_Shperes[-1,:].T.reshape(3*numShpere,1), - Index_Spheres[-1,:].T.reshape(3*numShpere,1)) \
                         - cdist(Ring_Spheres[1:4,:].T, Small_Spheres[1:4,:].T)

    L_Concise[numpy.where(L_Concise<0)]=0
    # print 'conflict', numpy.where(L_Concise>0)
    # L_Concise = numpy.max(L_Concise, numpy.zeros_like(L_Concise))

    Lambda =  1.0 / SubPixelNum
    term1 = Lambda*numpy.sum(SubD**2)
    term2 = numpy.sum(B**2)/totalSphere
    term3=numpy.sum(L_Concise**2)/7/numShpere/3

    Cost = term1 +term2 + term3
    # Cost = Lambda * numpy.sum(SubD**2) + numpy.sum(B**2) + numpy.sum(L_Concise**2)
    # print 'cost',Cost,'term 1, 2, 3',term1,term2,term3
    return Cost,term1,term2 ,term3
