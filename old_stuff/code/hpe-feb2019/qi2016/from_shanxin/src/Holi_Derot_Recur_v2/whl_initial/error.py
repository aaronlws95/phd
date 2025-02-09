__author__ = 'QiYE'
import h5py
import numpy
import scipy.io

from src.utils import constants
from src.utils.err_uvd_xyz import err_in_ori_xyz


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

    whole_err_uvd_xyz(dataset='test',
                      dataset_path_prefix=constants.Data_Path,
                      setname='icvl',
                souce_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                pred_save_name = '_uvd_bw_r012_21jnts_64_96_128_1_2_adam_lm9')

    # whole_err_uvd_xyz(dataset='train',
    #                   dataset_path_prefix=constants.Data_Path,
    #                   setname='msrc',
    #             souce_name_ori='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    #             pred_save_name = '_whole_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm300_yt0_ep340')
