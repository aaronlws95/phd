__author__ = 'QiYE'

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
import numpy
from . import LabelUtils
import h5py
from pyswarm import pso
from ..utils import xyz_uvd
from scipy.spatial.distance import cdist

#####Hand Model Parameter for msrc test######
pointCloudMargin=50
palmBaseTopScaleRatio=0.6
numInSphere=4
numInSphereThumb1=4
FocalLength = 475.0659
setname='mega'
SubPixelNum=2048
ImgCenter=[320,240]
jnt_idx=range(0,21,1)

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
    # SubPixelNum=PixelInd[0].shape[0]
    # SubPixelInd =(PixelInd[0],PixelInd[1])

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
    return Cost





def cost_function_pso(x,*args):
    setname,DepthImg, inSpheres, Center, SilhouetteDistImg,SubPixelNum,perpend=args


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
        SubPixelNum = PixelInd[0].shape[0]-1
    tmp = numpy.random.randint(0,PixelInd[0].shape[0],SubPixelNum)
    SubPixelInd =(PixelInd[0][tmp],PixelInd[1][tmp])

    # SubPixelNum=PixelInd[0].shape[0]
    # SubPixelInd =(PixelInd[0],PixelInd[1])


    # SubD = numpy.zeros(1, SubPixelNum)
    # DepthDiscrepancyThreshold = 10
    Locates = numpy.empty((SubPixelNum,3))

    Locates[:,0]=Gx[SubPixelInd]
    Locates[:,1]=Gy[SubPixelInd]
    Locates[:,2]=Gz[SubPixelInd]
    # ShowPointCloud(points0=Locates,hand_points=Spheres[1:4,SLICE_IDX_SKE_FROM_SPHERE].T,Sphere=Spheres.T)
    dist_p_s = cdist(Locates, Spheres[1:4, :].T)
    closest_sphere = numpy.argmin(dist_p_s,axis=-1)
    closest_dist = dist_p_s[:,closest_sphere]
    tmp=numpy.ones_like(dist_p_s)*(Spheres[4,:].reshape(1,totalSphere))
    closest_sphere_radius = tmp[:,closest_sphere]
    Cost_D = numpy.abs(closest_dist-closest_sphere_radius)


    # Dists = numpy.abs( -Spheres[4,:].reshape(1,totalSphere))
    # Dists = numpy.abs(cdist(Locates, Spheres[1:4, :].T) - numpy.ones((SubPixelNum, Spheres.shape[-1]))*Spheres[4,:].T)
    # SubD = Dists.T
    # SubD = numpy.min(Dists.T,axis=0)
    # print SubD.shape

    # B = numpy.zeros((1, totalSphere))
    # sphereUVD = xyz_uvd.xyz2uvd(setname=setname,xyz=Spheres[1:4,:].T)
    # u = numpy.asarray(numpy.round(sphereUVD[:,0]),dtype='int16')
    # v =  numpy.asarray(numpy.round(sphereUVD[:,1]),dtype='int16')
    #
    # # % check whether u or v is out of the range
    # if (max(u) >= Center[0]*2) or (min(u) <= 0) or (max(v) >= Center[1]*2) or (min(v) <= 0):
    #     B = 1000 * numpy.ones((1,totalSphere))
    # else:
    #     DepthProj = DepthImg[(v,u)]
    #     DepthSphere = sphereUVD[:,2]
    #     # % Find the valid projected point
    #     ValidSpheresProjInd = numpy.where(DepthProj>0)
    #     InValidSpheresProjInd = numpy.where(0 == DepthProj)
    #     # templength = ValidSpheresProjInd[0].shape[0]
    #     temp1 = DepthProj[ValidSpheresProjInd] - DepthSphere[ValidSpheresProjInd]
    #     temp1[numpy.where(temp1<0)]=0
    #     # temp2 = numpy.max([numpy.zeros((templength,)), temp1],axis=0)
    #     # %B(ValidSpheresProjInd) = min([DepthDiscrepancyThreshold*ones(1,templength) temp2])
    #     B[:,ValidSpheresProjInd]= temp1
    #     invalidVU = (v[InValidSpheresProjInd],u[InValidSpheresProjInd])
    #     B[:,InValidSpheresProjInd] = SilhouetteDistImg[invalidVU]
    #     # print 'InValidSpheresProjInd',InValidSpheresProjInd, SilhouetteDistImg[invalidVU]
    #     # B[InValidSpheresProjInd] = SilhouetteDistImg(indices(InValidSpheresProjInd))


    # Lambda =  1.0 / SubPixelNum
    # term1 = Lambda*numpy.sum(Cost_D**2)
    # term2 = numpy.sum(B**2)/totalSphere
    #
    # Cost = term1 +term2
    # # Cost = Lambda * numpy.sum(SubD**2) + numpy.sum(B**2)
    # print('cost',Cost)
    # Lambda =  1.0 / SubPixelNum
    return numpy.mean(Cost_D)




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
    uvd,xyz,file_name=read_dataset_file(save_dir)

    Hand_Sphere_Raidus=numpy.array([ 40,
      36,  28,  22,   20,
      34,   26,  22,   20,
      34,   26,  22,   20,
      34,   26,  22,   20,
      34,  26,  22,   20])/2.0

    for i in range(14,uvd.shape[0],1):
        print(i,file_name[i])
        roiDepth =Image.open("%s/images/%s"%(img_dir,file_name[i]))
        depth = numpy.asarray(roiDepth, dtype='uint32')
        cur_xyz=xyz[i]
        cur_uvd=uvd[i]


        sphere = LabelUtils.skeleton2sphere_21jnt(Skeleton=cur_xyz, numInSphere=numInSphere,
                                 palmBaseTopScaleRatio=palmBaseTopScaleRatio,Hand_Sphere_Raidus=Hand_Sphere_Raidus)

        handOnlyDepth = getHandOnlyImg(depth=depth,hand_jnt_gt_uvd=cur_uvd)
        silDistImg = getSilhouetteDistImg(handOnlyDepth,FocalLength)

        # line0= cur_xyz[5]-cur_xyz[0]
        # line0 /= numpy.sqrt(numpy.sum(line0**2))
        # line1=cur_xyz[13]-cur_xyz[0]
        # line1 /= numpy.sqrt(numpy.sum(line1**2))
        # perpend = numpy.cross(line0,line1)
        # print(perpend)

        all_cost=[]
        max_offset = 20
        step = 3
        all_offset = [0]+ list(range(step,max_offset,step))+ list(range(-step,max_offset,-step))
        offset_xyz=[]
        for i in all_offset:
            for j in all_offset:
                for k in all_offset:
                    inSpheres=sphere.copy()
                    inSpheres[:,:,1]+=i
                    inSpheres[:,:,2]+=j
                    inSpheres[:,:,3]+=k
                    offset_xyz.append([i,j,k])
                    cost= cost_function(setname=setname,DepthImg=handOnlyDepth, inSpheres=inSpheres, Center=ImgCenter,
                                         SilhouetteDistImg=silDistImg,SubPixelNum=SubPixelNum)
                    all_cost.append(cost)
        loc = numpy.argmin(all_cost)

        print(loc)
        xopt=offset_xyz[loc]
        print(xopt,all_cost[loc],'oricost',all_cost[0])
        title='%d,%s'%(i,file_name[i])
        inSpheres=sphere.copy()
        inSpheres[:,:,1]+=xopt[0]
        inSpheres[:,:,2]+=xopt[1]
        inSpheres[:,:,3]+=xopt[2]
                # cur_xyz+=xopt

        selectShpere=inSpheres[jnt_idx]
        uvd_point_label, partMap,ColorMap = LabelUtils.getPart_mega_21jnt_tmptest(setname=setname,DepthImg=depth,
                                     Sphere=selectShpere.reshape(len(jnt_idx)*numInSphere,5),numInSphere=numInSphere,
                                     numJnt=len(jnt_idx),pointCloudMargin=pointCloudMargin)

        # ShowDepthSphere(title=title,setname=setname,depth=depth,Sphere=sphere)
        LabelUtils.ShowDepthSphereMaskHist2(title=title,setname=setname,depth=depth,numPoint=None,
                                            jnt_uvd=cur_uvd,hand_points=cur_xyz,Sphere=selectShpere,partMap=partMap)
        #
        selectShpere=sphere[jnt_idx]
        uvd_point_label, partMap,ColorMap = LabelUtils.getPart_mega_21jnt_tmptest(setname=setname,DepthImg=depth,
                                     Sphere=selectShpere.reshape(len(jnt_idx)*numInSphere,5),numInSphere=numInSphere,
                                     numJnt=len(jnt_idx),pointCloudMargin=pointCloudMargin)

        # ShowDepthSphere(title=title,setname=setname,depth=depth,Sphere=sphere)
        LabelUtils.ShowDepthSphereMaskHist2(title=title,setname=setname,depth=depth,numPoint=None,jnt_uvd=cur_uvd,
                                            hand_points=cur_xyz,Sphere=selectShpere,partMap=partMap)
        #

if __name__=='__main__':
    evalaute_mega_online()