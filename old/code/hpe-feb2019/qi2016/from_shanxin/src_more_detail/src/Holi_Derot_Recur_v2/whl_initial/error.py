__author__ = 'QiYE'
import h5py
import numpy
import scipy.io
import matplotlib.pyplot as plt
from src.utils import constants
from src.utils.err_uvd_xyz import err_in_ori_xyz
from src.utils import convert

from src.utils import xyz_uvd

def normuvd2xyz_mega(path,cur_norm_uvd):
    f = h5py.File(path,'r')
    xyz_jnt_gt = f['xyz_jnt_gt'][...]
    uvd_norm = f['uvd_jnt_gt_norm'][...]
    uvd_jnt_gt =f['uvd_jnt_gt']
    bbsize = f['bbsize'][...]
    ref_z = f['ref_z'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    resxy = f['resxy'][...]
    r0=f['r0'][...]
    f.close()

    cur_norm_uvd=cur_norm_uvd-numpy.array([0.5,0.5,0.5])

    bb = numpy.array([(bbsize,bbsize,ref_z)])

    u0= 315.944855
    v0= 245.287079
    bbox_uvd = convert.xyz2uvd(xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] - u0)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - v0)

    mean_u = hand_center_uvd[:,0]
    mean_v = hand_center_uvd[:,1]
    mean_z = hand_center_uvd[:,2]

    cur_uvd = numpy.empty_like(cur_norm_uvd)
    cur_uvd[:,:,0] = cur_norm_uvd[:,:,0] *bbox_uvd[0,0]+mean_u.reshape(mean_u.shape[0],1)
    cur_uvd[:,:,1] = cur_norm_uvd[:,:,1] *bbox_uvd[0,1]+mean_v.reshape(mean_v.shape[0],1)
    cur_uvd[:,:,2] = cur_norm_uvd[:,:,2]*bbsize+ref_z

    xyz = convert.uvd2xyz(uvd=cur_uvd)
    xyz[:,:,2] = xyz[:,:,2] - ref_z+mean_z.reshape(mean_z.shape[0],1)
    err =numpy.sqrt(numpy.sum((xyz-xyz_jnt_gt)**2,axis=-1))

    print 'mean err ', numpy.mean(numpy.mean(err))
    print 'mean err for 21 joints', numpy.mean(err,axis=0)
    abs_err =numpy.abs(xyz-xyz_jnt_gt)

    return xyz,xyz_jnt_gt, err



def normuvd2xyz_Shanxin(setname,jnt_idx,path,cur_norm_uvd):
    f = h5py.File(path,'r')
    xyz_jnt_gt = f['xyz_jnt_gt'][...][:,jnt_idx,:]
    bbsize = f['bbsize'][...]
    ref_z = f['ref_z'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    resxy = f['resxy'][...]
    r0=f['r0'][...]
    f.close()

    cur_norm_uvd=cur_norm_uvd-numpy.array([0.5,0.5,0.5])

    u0= resxy[0]/2
    v0= resxy[1]/2
    bb = numpy.array([(bbsize,bbsize,ref_z)])

    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] -u0)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - v0)



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
    return xyz,xyz_jnt_gt,err



def normuvd2xyz(setname,path,cur_norm_uvd):
    f = h5py.File(path,'r')
    xyz_jnt_gt = f['xyz_jnt_gt'][...]
    bbsize = f['bbsize'][...]
    ref_z = f['ref_z'][...]
    hand_center_uvd = f['hand_center_uvd'][...]
    resxy = f['resxy'][...]
    r0=f['r0'][...]
    f.close()

    cur_norm_uvd=cur_norm_uvd-numpy.array([0.5,0.5,0.5])

    u0= resxy[0]/2
    v0= resxy[1]/2
    bb = numpy.array([(bbsize,bbsize,ref_z)])

    bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
    bbox_uvd[0,0] = numpy.ceil(bbox_uvd[0,0] -u0)
    bbox_uvd[0,1] = numpy.ceil(bbox_uvd[0,1] - v0)



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

    # print 'mean err ', numpy.mean(numpy.mean(err))
    # print 'mean err for 21 joints', numpy.mean(err,axis=0)


    #
    # for i in numpy.random.randint(0,cur_uvd.shape[0],20):
    #
    #     plt.imshow(r0[i],'gray')
    #     cur_uvd = cur_norm_uvd[i]
    #     plt.scatter(cur_uvd[:,0]*96+48,cur_uvd[:,1]*96+48)
    #     print err[i]
    #     print numpy.mean(err[i])
    #     # print numpy.mean(abs_err[i,:,0])
    #     # print numpy.mean(abs_err[i,:,1])
    #     # print numpy.mean(abs_err[i,:,2])
    #     plt.show()

    return xyz,xyz_jnt_gt,err

def whole_err_uvd_xyz(dataset,dataset_path_prefix,souce_name_ori,setname,pred_save_name):
    jnt_idx = range(0,21,1)
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,souce_name_ori)
    f = h5py.File(path,'r')
    r0=f['r0'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]

    orig_pad_border=f['orig_pad_border'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    drange = depth_dmin_dmax[:,1]-depth_dmin_dmax[:,0]
    pred_path = '%sdata/%s/holi_derot_recur_v2/whl_initial/'%(dataset_path_prefix,setname)
    path = '%s%s%s.npy'%(pred_path,dataset,pred_save_name)
    whole_pred = numpy.load(path)
    whole_pred.shape=(whole_pred.shape[0],21,3)

    err_uvd = numpy.mean(numpy.sqrt(numpy.sum((whole_pred -joint_label_uvd)**2,axis=-1)),axis=0)
    print 'norm error', err_uvd
    # for i in numpy.random.randint(0,r0.shape[0],5):
    #     plt.figure()
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(whole_pred[i,:,0]*72+12,whole_pred[i,:,1]*72+12)
    #     plt.scatter(joint_label_uvd[i,:,0]*72+12,joint_label_uvd[i,:,1]*72+12,c='r')
    #     plt.show()

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']


    xyz_pred = err_in_ori_xyz(setname,whole_pred,joint_label_uvd,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)
    # direct ='%sdata/%s/holi_derot_recur_v2/whl_initial/'%(dataset_path_prefix,setname)
    # numpy.save("%s%s_xyz%s.npy"%(direct,dataset,pred_save_name),xyz_pred)


if __name__=='__main__':
    # whole_err_uvd_xyz(dataset='train',setname='nyu',dataset_path_prefix=constants.Data_Path,
    #             souce_name_ori='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             pred_save_name = '_whole_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm300_yt0_ep905')

    # whole_err_uvd_xyz(dataset='test',
    #                   dataset_path_prefix=constants.Data_Path,
    #                   setname='icvl',
    #             souce_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             pred_save_name = '_uvd_bw_r012_21jnts_64_96_128_1_2_adam_lm9')
    #
    # # whole_err_uvd_xyz(dataset='train',
    # #                   dataset_path_prefix=constants.Data_Path,
    # #                   setname='msrc',
    # #             souce_name_ori='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    # #             pred_save_name = '_whole_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm300_yt0_ep340')
    #

    dataset = 'test'
    setname='mega'
    dataset_path_prefix='/home/icvl/Qi/Prj_CNN_Hier_NewData/'
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s/test_norm_hand_uvd_rootmid.h5'%src_path
    normuvd2xyz(setname,path)
