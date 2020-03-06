__author__ = 'QiYE'
import numpy
import h5py
import matplotlib.pyplot as plt

from src.SphereHandModel.utils import xyz_uvd


def normuvd2xyz_err_neutral(setname,path,cur_norm_uvd):
    idx = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
    f = h5py.File(path,'r')
    xyz_jnt_gt = f['xyz_jnt_gt'][...][:,idx,:]
    bbsize1 = f['bbsize1'][...]
    ref_z = f['ref_z'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    resxy = f['resxy'][...]
    # r0=f['r0'][...]
    f.close()

    resx= resxy[0]
    resy = resxy[1]

    # for i in xrange(2500,5200,20):
    #
    #     plt.imshow(r0[i],'gray')
    #     cur_uvd = cur_norm_uvd[i].reshape(20,3)
    #     plt.scatter(cur_uvd[:,0]*96+48,cur_uvd[:,1]*96+48)
    #     plt.show()

    mean_u = hand_center_uvd[:,0]
    mean_v = hand_center_uvd[:,1]
    mean_z = hand_center_uvd[:,2]

    cur_uvd = numpy.empty_like(cur_norm_uvd)
    start_idx_person_2 = xyz_jnt_gt.shape[0]

    bbsize=bbsize1
    bb = numpy.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - resx/2)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - resy/2)

    cur_uvd[:,:,0] = cur_norm_uvd[:,:,0] *bbox_uvd[0,0]+mean_u[:].reshape(start_idx_person_2,1)
    cur_uvd[:,:,1] = cur_norm_uvd[:,:,1] *bbox_uvd[0,1]+mean_v[:].reshape(start_idx_person_2,1)
    cur_uvd[:,:,2] = cur_norm_uvd[:,:,2]*bbsize+ref_z


    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=cur_uvd)
    xyz[:,:,2] = xyz[:,:,2] - ref_z+mean_z.reshape(mean_z.shape[0],1)
    err =numpy.sqrt(numpy.sum((xyz-xyz_jnt_gt)**2,axis=-1))

    print 'err mean', numpy.mean(numpy.mean(err))
    return xyz,xyz_jnt_gt,err

def normuvd2xyz_err_neutral_scale(setname,path,cur_norm_uvd):
    idx = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
    f = h5py.File(path,'r')
    xyz_jnt_gt = f['xyz_jnt_gt'][...][:,idx,:]
    bbsize1 = f['bbsize1'][...]
    bbsize2 = f['bbsize2'][...]
    ref_z = f['ref_z'][...]
    # r0 = f['r0'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    resxy = f['resxy'][...]
    f.close()

    resx= resxy[0]
    resy = resxy[1]


    # for i in xrange(2500,5200,20):
    #
    #     plt.imshow(r0[i],'gray')
    #     cur_uvd = cur_norm_uvd[i].reshape(20,3)
    #     plt.scatter(cur_uvd[:,0]*96+48,cur_uvd[:,1]*96+48)
    #     plt.show()

    mean_u = hand_center_uvd[:,0]
    mean_v = hand_center_uvd[:,1]
    mean_z = hand_center_uvd[:,2]

    cur_uvd = numpy.empty_like(cur_norm_uvd)
    start_idx_person_2 = 2440
    neutral_num = 8252

    bbsize=bbsize1
    bb = numpy.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - resx/2)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - resy/2)
    person_1_frames = range(0,start_idx_person_2,1)
    cur_uvd[person_1_frames,:,0] = cur_norm_uvd[person_1_frames,:,0] *bbox_uvd[0,0]+mean_u[person_1_frames].reshape(start_idx_person_2,1)
    cur_uvd[person_1_frames,:,1] = cur_norm_uvd[person_1_frames,:,1] *bbox_uvd[0,1]+mean_v[person_1_frames].reshape(start_idx_person_2,1)
    cur_uvd[person_1_frames,:,2] = cur_norm_uvd[person_1_frames,:,2]*bbsize+ref_z


    bbsize=bbsize2
    bb = numpy.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - resx/2)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - resy/2)
    person_2_frames = range(start_idx_person_2,cur_norm_uvd.shape[0],1)
    cur_uvd[person_2_frames,:,0] = cur_norm_uvd[person_2_frames,:,0] *bbox_uvd[0,0]+mean_u[person_2_frames].reshape(neutral_num-start_idx_person_2,1)
    cur_uvd[person_2_frames,:,1] = cur_norm_uvd[person_2_frames,:,1] *bbox_uvd[0,1]+mean_v[person_2_frames].reshape(neutral_num-start_idx_person_2,1)
    cur_uvd[person_2_frames,:,2] = cur_norm_uvd[person_2_frames,:,2]*bbsize+ref_z


    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=cur_uvd)
    xyz[:,:,2] = xyz[:,:,2] - ref_z+mean_z.reshape(mean_z.shape[0],1)
    err =numpy.sqrt(numpy.sum((xyz-xyz_jnt_gt)**2,axis=-1))


    print 'err neutural tomposn', numpy.mean(numpy.mean(err[0:start_idx_person_2]))
    print 'err neutural the other', numpy.mean(numpy.mean(err[2440:neutral_num]))
    print 'all', numpy.mean(numpy.mean(err))

    jnt11_idx = [0, 2, 6, 9, 13, 17, 4, 8, 11, 15, 19]
    print '11 jnts', numpy.mean(numpy.mean(err[:,jnt11_idx]))
    return xyz,xyz_jnt_gt,err
def normuvd2xyz_err_neutral_scale2(setname,path,cur_norm_uvd,idx):
    # idx = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
    f = h5py.File(path,'r')
    xyz_jnt_gt = f['xyz_jnt_gt'][...][:,idx,:]
    bbsize1 = f['bbsize1'][...]
    bbsize2 = f['bbsize2'][...]
    ref_z = f['ref_z'][...]
    # r0 = f['r0'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    resxy = f['resxy'][...]
    f.close()

    resx= resxy[0]
    resy = resxy[1]


    # for i in xrange(2500,5200,20):
    #
    #     plt.imshow(r0[i],'gray')
    #     cur_uvd = cur_norm_uvd[i].reshape(20,3)
    #     plt.scatter(cur_uvd[:,0]*96+48,cur_uvd[:,1]*96+48)
    #     plt.show()

    mean_u = hand_center_uvd[:,0]
    mean_v = hand_center_uvd[:,1]
    mean_z = hand_center_uvd[:,2]

    cur_uvd = numpy.empty_like(cur_norm_uvd)
    start_idx_person_2 = 2440
    neutral_num = 8252

    bbsize=bbsize1
    bb = numpy.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - resx/2)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - resy/2)
    person_1_frames = range(0,start_idx_person_2,1)
    cur_uvd[person_1_frames,:,0] = cur_norm_uvd[person_1_frames,:,0] *bbox_uvd[0,0]+mean_u[person_1_frames].reshape(start_idx_person_2,1)
    cur_uvd[person_1_frames,:,1] = cur_norm_uvd[person_1_frames,:,1] *bbox_uvd[0,1]+mean_v[person_1_frames].reshape(start_idx_person_2,1)
    cur_uvd[person_1_frames,:,2] = cur_norm_uvd[person_1_frames,:,2]*bbsize+ref_z


    bbsize=bbsize2
    bb = numpy.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - resx/2)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - resy/2)
    person_2_frames = range(start_idx_person_2,cur_norm_uvd.shape[0],1)
    cur_uvd[person_2_frames,:,0] = cur_norm_uvd[person_2_frames,:,0] *bbox_uvd[0,0]+mean_u[person_2_frames].reshape(neutral_num-start_idx_person_2,1)
    cur_uvd[person_2_frames,:,1] = cur_norm_uvd[person_2_frames,:,1] *bbox_uvd[0,1]+mean_v[person_2_frames].reshape(neutral_num-start_idx_person_2,1)
    cur_uvd[person_2_frames,:,2] = cur_norm_uvd[person_2_frames,:,2]*bbsize+ref_z


    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=cur_uvd)
    xyz[:,:,2] = xyz[:,:,2] - ref_z+mean_z.reshape(mean_z.shape[0],1)
    err =numpy.sqrt(numpy.sum((xyz-xyz_jnt_gt)**2,axis=-1))


    print 'err neutural tomposn', numpy.mean(numpy.mean(err[0:start_idx_person_2]))
    print 'err neutural the other', numpy.mean(numpy.mean(err[2440:neutral_num]))
    print 'all', numpy.mean(numpy.mean(err))

    jnt11_idx = [0, 2, 6, 9, 13, 17, 4, 8, 11, 15, 19]
    print '11 jnts', numpy.mean(numpy.mean(err[:,jnt11_idx]))
    return xyz,xyz_jnt_gt,err
def normuvd2xyz_err_track_gt_scale(setname,path,norm_uvd,time_range,num_sample_frames):
    idx = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
    f = h5py.File(path,'r')
    xyz_jnt_gt = f['xyz_jnt_gt'][...][:,idx,:]
    uvd = f['uvd_jnt_gt_norm'][...][:,idx,:]
    bbsize1 = f['bbsize1'][...]
    bbsize2 = f['bbsize2'][...]
    ref_z = f['ref_z'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    resxy = f['resxy'][...]
    # r0=f['r0'][...]
    f.close()

    resx= resxy[0]
    resy = resxy[1]

    ori_num_sample = xyz_jnt_gt.shape[0]

    off_sample = ori_num_sample-time_range
    xyz_track = numpy.empty((num_sample_frames,ori_num_sample,20,3),dtype='float32')

    cur_norm_uvd = numpy.empty_like(xyz_jnt_gt)
    cur_norm_uvd[0:time_range] = uvd[0:time_range]
    print norm_uvd.shape
    err_seg = []
    for i in xrange(num_sample_frames):
        print 'frame seg', i
        cur_norm_uvd[time_range:ori_num_sample]  = norm_uvd[i*off_sample:(i+1)*off_sample]

        # # for si in xrange(0,200,1):
        # for si in  numpy.random.randint(0, ori_num_sample, 5):
        #
        #     plt.imshow(r0[si],'gray')
        #     tmp = cur_norm_uvd[si].reshape(20,3)
        #     plt.scatter(tmp[:,0]*96,tmp[:,1]*96)
        #     plt.show()

        mean_u = hand_center_uvd[:,0]
        mean_v = hand_center_uvd[:,1]
        mean_z = hand_center_uvd[:,2]

        cur_uvd = numpy.empty_like(cur_norm_uvd)
        start_idx_person_2 = 2440
        neutral_num = 8252

        bbsize=bbsize1
        bb = numpy.array([(bbsize,bbsize,ref_z)])
        bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
        bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - resx/2)
        bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - resy/2)
        person_1_frames = range(0,start_idx_person_2,1)
        cur_uvd[person_1_frames,:,0] = cur_norm_uvd[person_1_frames,:,0] *bbox_uvd[0,0]+mean_u[person_1_frames].reshape(start_idx_person_2,1)
        cur_uvd[person_1_frames,:,1] = cur_norm_uvd[person_1_frames,:,1] *bbox_uvd[0,1]+mean_v[person_1_frames].reshape(start_idx_person_2,1)
        cur_uvd[person_1_frames,:,2] = cur_norm_uvd[person_1_frames,:,2]*bbsize+ref_z


        bbsize=bbsize2
        bb = numpy.array([(bbsize,bbsize,ref_z)])
        bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
        bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - resx/2)
        bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - resy/2)
        person_2_frames = range(start_idx_person_2,cur_norm_uvd.shape[0],1)
        cur_uvd[person_2_frames,:,0] = cur_norm_uvd[person_2_frames,:,0] *bbox_uvd[0,0]+mean_u[person_2_frames].reshape(neutral_num-start_idx_person_2,1)
        cur_uvd[person_2_frames,:,1] = cur_norm_uvd[person_2_frames,:,1] *bbox_uvd[0,1]+mean_v[person_2_frames].reshape(neutral_num-start_idx_person_2,1)
        cur_uvd[person_2_frames,:,2] = cur_norm_uvd[person_2_frames,:,2]*bbsize+ref_z


        xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=cur_uvd)
        xyz[:,:,2] = xyz[:,:,2] - ref_z+mean_z.reshape(mean_z.shape[0],1)
        err =numpy.sqrt(numpy.sum((xyz-xyz_jnt_gt)**2,axis=-1))


        print 'err neutural tomposn', numpy.mean(numpy.mean(err[0:start_idx_person_2]))
        print 'err neutural the other', numpy.mean(numpy.mean(err[start_idx_person_2:neutral_num]))
        jnt11_idx = [0, 2, 6, 9, 13, 17, 4, 8, 11, 15, 19]
        print '11 jnts', numpy.mean(numpy.mean(err[:,jnt11_idx]))
        print 'all', numpy.mean(numpy.mean(err))

        xyz_track[i]=xyz
        err_seg.append(numpy.mean(numpy.mean(err)))
    print 'err in different seg', err_seg
    print 'mean', numpy.mean(err_seg)
    return xyz_track,xyz_jnt_gt


def normuvd2xyz(setname,path,cur_norm_uvd):
    idx = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20]
    f = h5py.File(path,'r')
    xyz_jnt_gt = f['xyz_jnt_gt'][...][:,idx,:]
    uvd_norm = f['uvd_jnt_gt'][...][:,idx,:]
    bbsize = f['bbsize'][...]
    ref_z = f['ref_z'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    resxy = f['resxy'][...]
    r0=f['hand'][...]
    f.close()

    resx= resxy[0]
    resy = resxy[1]
    bb = numpy.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb,jnt_type='single' )
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - resx/2)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - resy/2)

    for i in xrange(100,200,20):

        plt.imshow(r0[i],'gray')
        cur_uvd = cur_norm_uvd[i].reshape(20,3)
        plt.scatter(cur_uvd[:,0]*96+48,cur_uvd[:,1]*96+48)
        plt.show()
    mean_u = hand_center_uvd[:,0]
    mean_v = hand_center_uvd[:,1]
    mean_z = hand_center_uvd[:,2]

    cur_uvd = numpy.empty_like(cur_norm_uvd)
    cur_uvd[:,:,0] = cur_norm_uvd[:,:,0] *bbox_uvd[0,0]+mean_u.reshape(mean_u.shape[0],1)
    cur_uvd[:,:,1] = cur_norm_uvd[:,:,1] *bbox_uvd[0,1]+mean_v.reshape(mean_v.shape[0],1)
    cur_uvd[:,:,2] = cur_norm_uvd[:,:,2]*bbsize+ref_z

    xyz = xyz_uvd.uvd2xyz(setname=setname,uvd=cur_uvd)
    xyz[:,:,2] = xyz[:,:,2] - ref_z+mean_z.reshape(mean_z.shape[0],1)
    err =numpy.sqrt(numpy.sum((xyz-xyz_jnt_gt)**2,axis=-1))

    print 'mean err ', numpy.mean(numpy.mean(err))
    print 'mean err for 21 joints', numpy.mean(numpy.mean(err))

    return xyz
