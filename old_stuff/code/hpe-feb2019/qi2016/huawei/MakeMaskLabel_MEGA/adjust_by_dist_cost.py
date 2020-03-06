import numpy
from ..utils import xyz_uvd
from scipy.spatial.distance import cdist
# from src.SphereHandModel.ShowSamples import *

def cost_function(setname,DepthImg, inSpheres, Center, SilhouetteDistImg,SubPixelNum):

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

    dist_p_s = cdist(Locates, Spheres[1:4, :].T)
    closest_sphere = numpy.argmin(dist_p_s,axis=-1)
    closest_dist = dist_p_s[:,closest_sphere]
    tmp=numpy.ones_like(dist_p_s)*(Spheres[4,:].reshape(1,totalSphere))
    closest_sphere_radius = tmp[:,closest_sphere]
    Cost_D = numpy.abs(closest_dist-closest_sphere_radius)

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


    term1 = numpy.mean(Cost_D)
    term2 = numpy.sum(B)/totalSphere

    Cost = term1 +term2
    # Cost = Lambda * numpy.sum(SubD**2) + numpy.sum(B**2)
    # print('cost',Cost,'term 1, 2',term1,term2)
    return Cost,term1,term2

