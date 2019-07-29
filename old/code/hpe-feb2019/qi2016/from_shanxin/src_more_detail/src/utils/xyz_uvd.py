__author__ = 'QiYE'
import numpy
from matplotlib import pylab
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

def xyz2uvd(setname,xyz):

    if setname =='action' or setname =='ACTION':

        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor

    if setname =='msrc' or setname =='MSRC':

        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor
    if setname =='icvl' or setname =='ICVL':

        res_x = 320
        res_y = 240

        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
    if setname =='nyu'or setname =='NYU':

        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor

    uvd = numpy.empty_like(xyz)
    # print 'jnt_type',jnt_type
    # print 'roixy', roixy[0,0],roixy[0,2]
    if len(xyz.shape)==3:
        for i in xrange(xyz.shape[0]):

            trans_x= xyz[i,:,0]
            trans_y= xyz[i,:,1]
            trans_z = xyz[i,:,2]
            uvd[i,:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
            uvd[i,:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
            uvd[i,:,2] = trans_z #convert m to mm
    else:
        for i in xrange(xyz.shape[0]):
            trans_x= xyz[i,0]
            trans_y= xyz[i,1]
            trans_z = xyz[i,2]
            uvd[i,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
            uvd[i,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )
            uvd[i,2] = trans_z #convert m to mm
    return uvd

def uvd2xyz(setname,uvd):

    if setname =='msrc' or setname=='MSRC':
        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor
    if setname =='icvl' or setname =='ICVL':
        res_x = 320
        res_y = 240

        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
    if setname =='nyu' or setname=='NYU':
        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor
    # focal_length = numpy.sqrt(focal_length_x ^ 2 + focal_length_y ^ 2);
    xyz = numpy.empty_like(uvd)
    # print 'jnt_type',jnt_type
    # print 'roixy', roixy[0,0],roixy[0,2]
    if len(xyz.shape)==3:
        for i in xrange(xyz.shape[0]):

            z =  uvd[i,:,2]# convert mm to m
            xyz[i,:,2]=z
            xyz[i,:,0] = ( uvd[i,:,0]- res_x / 2)/res_x/ focal_length_x*z
            xyz[i,:,1] = ( uvd[i,:,1]- res_y / 2)/res_y/focal_length_y*z
    else:
        for i in xrange(xyz.shape[0]):

            z =  uvd[i,2]# convert mm to m
            xyz[i,2]=z
            xyz[i,0] = ( uvd[i,0]- res_x / 2)/res_x/ focal_length_x*z
            xyz[i,1] = ( uvd[i,1]- res_y / 2)/res_y/focal_length_y*z

    return xyz




def xyz2uvd_old_version(setname,xyz,roixy, jnt_type=None):
    if setname =='msrc' or setname =='MSRC':

        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor
    if setname =='icvl' or setname =='ICVL':

        res_x = 320
        res_y = 240

        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
    if setname =='nyu'or setname =='NYU':

        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor

    uvd = numpy.empty_like(xyz)
    # print 'jnt_type',jnt_type
    # print 'roixy', roixy[0,0],roixy[0,2]
    if jnt_type != 'single':

        for i in xrange(xyz.shape[0]):
            roi_xMin = roixy[i,0]
            roi_yMin = roixy[i,2]
            trans_x= xyz[i,:,0]
            trans_y= xyz[i,:,1]
            trans_z = xyz[i,:,2]
            uvd[i,:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )-roi_xMin
            uvd[i,:,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )-roi_yMin
            uvd[i,:,2] = trans_z*1000 #convert m to mm
    else:
        for i in xrange(xyz.shape[0]):
            roi_xMin = roixy[i,0]
            roi_yMin = roixy[i,2]
            trans_x= xyz[i,0]
            trans_y= xyz[i,1]
            trans_z = xyz[i,2]
            uvd[i,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )-roi_xMin
            uvd[i,1] = res_y / 2 + res_y * focal_length_y * ( trans_y / trans_z )-roi_yMin
            uvd[i,2] = trans_z*1000 #convert m to mm
    return uvd

def uvd2xyz_old_version(setname,uvd, roixy, jnt_type=None):

    if setname =='msrc' or setname=='MSRC':
        res_x = 512
        res_y = 424

        scalefactor = 1
        focal_length_x = 0.7129 * scalefactor
        focal_length_y =0.8608 * scalefactor
    if setname =='icvl' or setname =='ICVL':
        res_x = 320
        res_y = 240

        scalefactor = 1
        focal_length_x = 0.7531 * scalefactor
        focal_length_y =1.004  * scalefactor
    if setname =='nyu' or setname=='NYU':
        res_x = 640
        res_y = 480

        scalefactor = 1
        focal_length_x = 0.8925925 * scalefactor
        focal_length_y =1.190123339 * scalefactor
    # focal_length = numpy.sqrt(focal_length_x ^ 2 + focal_length_y ^ 2);
    xyz = numpy.empty_like(uvd)
    # print 'jnt_type',jnt_type
    # print 'roixy', roixy[0,0],roixy[0,2]
    if jnt_type != 'single':
        for i in xrange(xyz.shape[0]):
            roi_xMin = roixy[i,0]
            roi_yMin = roixy[i,2]
            z =  uvd[i,:,2]/1000 # convert mm to m
            xyz[i,:,2]=z
            xyz[i,:,0] = ( uvd[i,:,0]+roi_xMin - res_x / 2)/res_x/ focal_length_x*z
            xyz[i,:,1] = ( uvd[i,:,1] +roi_yMin- res_y / 2)/res_y/focal_length_y*z
    else:
        for i in xrange(xyz.shape[0]):
            roi_xMin = roixy[i,0]
            roi_yMin = roixy[i,2]
            z =  uvd[i,2]/1000 # convert mm to m
            xyz[i,2]=z
            xyz[i,0] = ( uvd[i,0] +roi_xMin- res_x / 2)/res_x/ focal_length_x*z
            xyz[i,1] = ( uvd[i,1] +roi_yMin- res_y / 2)/res_y/focal_length_y*z

    return xyz

# if __name__ == '__main__':
#     dataset = 'test'
#     keypoints = scipy.io.loadmat('C:/Proj/Proj_CNN_Hier/data/joint_uvd/uvd_msrc_%s_22joints.mat' %dataset)
#     uvd = keypoints['uvd']
#
#
#     keypoints = scipy.io.loadmat('C:/Proj/Proj_CNN_Hier/data/joint_uvd/xyz_msrc_%s_22joints.mat'%dataset)
#     xyz = keypoints['xyz']
#
#
#     new_xyz = uvd2xyz(uvd)
#     for i in xrange(16):
#         print xyz[0,i,:]
#         print new_xyz[0,i,:]