__author__ = 'QiYE'
import h5py
import numpy
import cv2
import scipy.io

from src.utils.err_uvd_xyz import err_in_ori_xyz
from src.utils import constants


def offset_to_abs(off_uvd, pre_uvd,patch_size=44,offset_depth_range=1.0,hand_width=96):

    if len(off_uvd.shape)<3:
        off_uvd[:,0:2] = (off_uvd[:,0:2]*patch_size -patch_size/2 )/hand_width
        # off_uvd[:,0:2] = (off_uvd[:,0:2]*72+12)/24
        predict_uvd= numpy.empty_like(off_uvd)
        predict_uvd[:,0:2] = pre_uvd[:,0:2]+off_uvd[:,0:2]
        off_uvd[:,2] = (off_uvd[:,2]-0.5)*offset_depth_range
        predict_uvd[:,2] = pre_uvd[:,2]+off_uvd[:,2]
        return predict_uvd
    else:
        pre_uvd.shape=(pre_uvd.shape[0],1,pre_uvd.shape[-1])
        off_uvd[:,:,0:2] = (off_uvd[:,:,0:2]*patch_size -patch_size/2 )/hand_width
        # off_uvd[:,0:2] = (off_uvd[:,0:2]*72+12)/24
        predict_uvd= numpy.empty_like(off_uvd)
        predict_uvd[:,:,0:2] = pre_uvd[:,:,0:2]+off_uvd[:,:,0:2]
        off_uvd[:,:,2] = (off_uvd[:,:,2]-0.5)*offset_depth_range
        predict_uvd[:,:,2] = pre_uvd[:,:,2]+off_uvd[:,:,2]
        return predict_uvd



def top_derot_err_uvd_xyz(dataset,prev_jnt_name,setname,dataset_path_prefix,source_name,source_name_ori,pred_save_name,jnt_idx,patch_size=24,offset_depth_range=0.8):

    src_path ='%sdata/%s/hier_derot_recur_v2/bw_offset/best/'%(constants.Data_Path,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)

    f = h5py.File(path,'r')
    rot = f['upd_rot'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    f.close()

    print 'index in whole hand',jnt_idx[0]
    direct =  '%sdata/%s/hier_derot_recur_v2/final_xyz_uvd/'%(dataset_path_prefix,setname)
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    print 'previous jnt name', prev_jnt_path
    prev_jnt_uvd_pred= numpy.load(prev_jnt_path)

    src_path ='%sdata/%s/source/'%(dataset_path_prefix,setname)

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']


    path = '%s%s%s.h5'%(src_path,dataset,source_name_ori)
    f = h5py.File(path,'r')
    uvd_gr = f['joint_label_uvd'][...]
    f.close()


    print prev_jnt_uvd_pred.shape

    direct =  '%sdata/%s/hier_derot_recur_v2/top/best/'%(dataset_path_prefix,setname)
    uvd_pred_offset =  numpy.load("%s%s%s.npy"%(direct,dataset,pred_save_name))
    print uvd_pred_offset.shape

    predict_uvd =uvd_pred_offset/10+prev_jnt_uvd_pred


    # direct ='%sdata/%s/hier_derot_recur_v2/final_xyz_uvd/'%(dataset_path_prefix,setname)
    # numpy.save("%s%s_absuvd0%s.npy"%(direct,dataset,pred_save_name),predict_uvd)
    """"rot the the norm view to original rotatioin view"""
    for i in xrange(uvd_gr.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot[i],1)

        predict_uvd[i,0:2] = (numpy.dot(M,numpy.array([predict_uvd[i,0]*96,predict_uvd[i,1]*96,1]))-12)/72

    xyz_pred,err = err_in_ori_xyz(setname,predict_uvd,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)
    # direct ='%sdata/%s/hier_derot_recur_v2/final_xyz_uvd/'%(dataset_path_prefix,setname)
    # numpy.save("%s%s_xyz%s.npy"%(direct,dataset,pred_save_name),xyz_pred)
    # numpy.save("%s%s_absuvd%s.npy"%(direct,dataset,pred_save_name),predict_uvd)
    return err


if __name__=='__main__':
    idx_all = [3,7,11,15,19]
    err_mean=[]
    for i , idx in enumerate(idx_all):
        err= top_derot_err_uvd_xyz(dataset='test',
                                   setname='icvl',
                                   dataset_path_prefix=constants.Data_Path,
                                 source_name='_iter1_whlimg_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                                 source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                                 prev_jnt_name='_absuvd0_offset_mid%d_patch40_beta6_24_48_1_1_adam_lm3'%(idx-1),
                                     pred_save_name='_offset_top%d_patch40_beta6_14_28_1_1_adam_lm3'%idx,
                                     patch_size=40,
                                     offset_depth_range=0.4,
                                     jnt_idx = [idx])
        err_mean.append(err)
    print err_mean
    print numpy.mean(err_mean)
    # idx=11
    # err= top_derot_err_uvd_xyz(dataset='test',
    #                            setname='icvl',
    #                            dataset_path_prefix=constants.Data_Path,
    #                          source_name='_iter1_whlimg_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          prev_jnt_name='_absuvd0_offset_mid%d_patch40_beta6_24_48_1_1_adam_lm3'%(idx-1),
    #                              pred_save_name='_offset_top%d_patch40_beta6_14_28_1_1_adam_lm3'%idx,
    #                              patch_size=40,
    #                              offset_depth_range=0.4,
    #                              jnt_idx = [idx])