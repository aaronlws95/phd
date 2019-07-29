__author__ = 'QiYE'
import h5py
import numpy
import cv2
import scipy.io
import matplotlib.pyplot as plt

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



def derot_err_uvd_xyz(num_iter,setname,dataset_path_prefix,source_name,source_name_ori,dataset,jnt_idx,pred_save_name ):
    if num_iter == 1:

        src_path = '%sdata/%s/holi_derot_recur_v2/whl_initial/'%(dataset_path_prefix,setname)
    else:

        src_path = '%sdata/%s/holi_derot_recur_v2/bw_offset/best/'%(dataset_path_prefix,setname)


    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    pred_uvd_derot = f['pred_uvd_derot'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]
    pred_uvd = f['pred_uvd'][...]
    gr_uvd = f['gr_uvd'][...]
    rot = f['upd_rot_iter1'][...]
    f.close()

    src_path ='%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name_ori)
    f = h5py.File(path,'r')
    # r0 = numpy.squeeze(f['r0'][...][jnt_idx])
    # r1 =  numpy.squeeze(f['r1'][...][jnt_idx])
    # r2=  numpy.squeeze(f['r2'][...][jnt_idx])
    uvd_gr = f['joint_label_uvd'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]

    f.close()

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' %(dataset_path_prefix,setname, dataset,setname))
    roixy = keypoints['roixy']

    print jnt_idx

    pred_uvd_derot.shape=(pred_uvd_derot.shape[0],21,3)
    pred_uvd.shape=(pred_uvd.shape[0],21,3)
    gr_uvd_derot.shape=(gr_uvd_derot.shape[0],21,3)
    gr_uvd.shape=(pred_uvd.shape[0],21,3)

    prev_jnt_uvd_derot = numpy.squeeze(pred_uvd_derot[:,jnt_idx,:])


    direct =  '%sdata/%s/holi_derot_recur_v2/bw_offset/'%(dataset_path_prefix,setname)
    uvd_pred_offset =  numpy.load("%s%s%s.npy"%(direct,dataset,pred_save_name))/10.0
    predict_uvd=uvd_pred_offset+prev_jnt_uvd_derot


    """"rot the the norm view to original rotatioin view"""
    for i in xrange(uvd_gr.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot[i],1)
        predict_uvd[i,0:2] = (numpy.dot(M,numpy.array([predict_uvd[i,0]*96,predict_uvd[i,1]*96,1]))-12)/72

    xyz_pred = err_in_ori_xyz(setname,predict_uvd,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)
    direct =  '%sdata/%s/holi_derot_recur_v2/final_xyz_uvd/'%(dataset_path_prefix,setname)
    # numpy.save("%s%s_xyz%s.npy"%(direct,dataset,pred_save_name),xyz_pred)
    numpy.save("%s%s_absuvd%s.npy"%(direct,dataset,pred_save_name),predict_uvd)

if __name__=='__main__':
    #
    # iter_absuvd_name=['_egoff_adam_iter1_bw0_r012_24_48_1_1_adam_lm29',
    #                    '_egoff_adam_iter1_bw1_r012_24_48_1_1_adam_lm0',
    #                    '_egoff_adam_iter1_bw5_r012_24_48_1_1_adam_lm3',
    #                    '_egoff_adam_iter1_bw9_r012_24_48_1_1_adam_lm29',
    #                    '_egoff_adam_iter1_bw13_r012_24_48_1_1_adam_lm3',
    #                    '_egoff_adam_iter1_bw17_r012_24_48_1_1_adam_lm3']
    # bw_idx = [0,1,5,9,13,17]
    # for i,idx in enumerate(bw_idx):
    #     derot_err_uvd_xyz(num_iter=1,
    #                   setname='icvl',
    #                   dataset_path_prefix=constants.Data_Path,
    #                              dataset='train',
    #                          source_name='_iter0_whlimg_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                              pred_save_name=iter_absuvd_name[i],
    #                                  jnt_idx = [idx])
    derot_err_uvd_xyz(num_iter=2,
                  setname='icvl',
                  dataset_path_prefix=constants.Data_Path,
                             dataset='test',
                         source_name='_iter1_whlimg_holi_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                         source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                             pred_save_name='_egoff_iter2_bw9_r012_24_48_1_1_adam_lm99',
                                 jnt_idx = [9])

