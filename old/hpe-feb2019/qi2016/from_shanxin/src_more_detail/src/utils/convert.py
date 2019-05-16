__author__ = 'QiYE'
import numpy
from matplotlib import pylab


def convert_depth_to_uvd_Shanxin(depth):
    v, u = pylab.meshgrid(range(0, depth.shape[0], 1), range(0, depth.shape[1], 1), indexing= 'ij')
    v = numpy.asarray(v, 'uint16')[:, :, numpy.newaxis]
    u = numpy.asarray(u, 'uint16')[:, :, numpy.newaxis]
    depth = depth[:, :, numpy.newaxis]
    uvd = numpy.concatenate((u, v, depth), axis=2)
    # print v.shape,u.shape,uvd.shape
    return uvd


def convert_depth_to_uvd(depth):
    v, u = pylab.meshgrid(range(0, depth.shape[0], 1), range(0, depth.shape[1], 1), indexing= 'ij')
    # print v[0,0:10]
    # print u[0,0:10]
    v = numpy.asarray(v, 'uint16')[:, :, numpy.newaxis]
    u = numpy.asarray(u, 'uint16')[:, :, numpy.newaxis]
    depth = depth[:, :, numpy.newaxis]
    uvd = numpy.concatenate((u, v, depth), axis=2)

    # print v.shape,u.shape,uvd.shape
    return uvd


def uvd2xyz(uvd):
    focal_length_x = 475.065948
    focal_length_y = 475.065857
    u0= 315.944855
    v0= 245.287079

    if len(uvd.shape)==3:
        xyz = numpy.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
        xyz[:,:,2]=uvd[:,:,2]
        xyz[:,:,0] = ( uvd[:,:,0] - u0)/focal_length_x*xyz[:,:,2]
        xyz[:,:,1] = ( uvd[:,:,1]- v0)/focal_length_y*xyz[:,:,2]
    else:
        xyz = numpy.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
        z =  uvd[:,2] # convert mm to m
        xyz[:,2]=z
        xyz[:,0] = ( uvd[:,0]- u0)/focal_length_x*z
        xyz[:,1] = ( uvd[:,1]- v0)/focal_length_y*z
    return xyz

def xyz2uvd(xyz):
    focal_length_x = 475.065948
    focal_length_y = 475.065857
    u0= 315.944855
    v0= 245.287079

    uvd = numpy.empty_like(xyz)
    if len(uvd.shape)==3:
        trans_x= xyz[:,:,0]
        trans_y= xyz[:,:,1]
        trans_z = xyz[:,:,2]
        uvd[:,:,0] = u0 +focal_length_x * ( trans_x / trans_z )
        uvd[:,:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
        uvd[:,:,2] = trans_z #convert m to mm
    else:
        trans_x= xyz[:,0]
        trans_y= xyz[:,1]
        trans_z = xyz[:,2]
        uvd[:,0] = u0 +  focal_length_x * ( trans_x / trans_z )
        uvd[:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
        uvd[:,2] = trans_z #convert m to mm
    return uvd




def uvd2xyz_Shanxin(uvd, fx, fy, uc, vc):
    focal_length_x = fx
    focal_length_y = fy
    u0= uc
    v0= vc

    if len(uvd.shape)==3:
        xyz = numpy.empty((uvd.shape[0],uvd.shape[1],uvd.shape[2]),dtype='float32')
        xyz[:,:,2]=uvd[:,:,2]
        xyz[:,:,0] = ( uvd[:,:,0] - u0)/focal_length_x*xyz[:,:,2]
        xyz[:,:,1] = ( uvd[:,:,1] - v0)/focal_length_y*xyz[:,:,2]
    else:
        xyz = numpy.empty((uvd.shape[0],uvd.shape[1]),dtype='float32')
        z =  uvd[:,2] # convert mm to m
        xyz[:,2]=z
        xyz[:,0] = ( uvd[:,0]- u0)/focal_length_x*z
        xyz[:,1] = ( uvd[:,1]- v0)/focal_length_y*z
    return xyz




def xyz2uvd_Shanxin(xyz,fx, fy, uc, vc):
    focal_length_x = fx
    focal_length_y = fy
    u0= uc
    v0= vc

    uvd = numpy.empty_like(xyz)
    if len(uvd.shape)==3:
        trans_x= xyz[:,:,0]
        trans_y= xyz[:,:,1]
        trans_z = xyz[:,:,2]
        uvd[:,:,0] = u0 +  focal_length_x * ( trans_x / trans_z )
        uvd[:,:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
        uvd[:,:,2] = trans_z #convert m to mm
    else:
        trans_x= xyz[:,0]
        trans_y= xyz[:,1]
        trans_z = xyz[:,2]
        uvd[:,0] = u0 +  focal_length_x * ( trans_x / trans_z )
        uvd[:,1] = v0 +  focal_length_y * ( trans_y / trans_z )
        uvd[:,2] = trans_z #convert m to mm
    return uvd


