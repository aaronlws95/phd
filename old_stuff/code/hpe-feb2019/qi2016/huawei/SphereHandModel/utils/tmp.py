__author__ = 'QiYE'

import h5py


path_train =  'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\NYU_dataset\\NYU_dataset\\train_norm_hand\\norm_hand_uvd_rootmid'

f = h5py.File('F:/Proj_CNN_tracking/data/source/train_norm_hand/norm_hand_uvd_rootmid.h5', 'r')
norm_hand_uvd =f['hand'][...]
uvd_jnt_gt =f['uvd_jnt_gt'][...]
xyz_jnt_gt =f['xyz_jnt_gt'][...]
hand_center_uvd =f['hand_center_uvd'][...]
roixy =f['roixy'][...]
resxy =f['resxy'][...]
ref_z =f['ref_z'][...]
bbsize =f['bbsize'][...]
f.close()


f = h5py.File('F:/Proj_CNN_tracking/data/source/train_norm_hand/norm_hand_uvd_rootmid_scale.h5', 'w')

f.create_dataset('r0', data=norm_hand_uvd)
f.create_dataset('xyz_jnt_gt', data=xyz_jnt_gt)
f.create_dataset('uvd_jnt_gt_norm', data=uvd_jnt_gt)
f.create_dataset('hand_center_uvd', data=hand_center_uvd)
f.create_dataset('resxy', data=resxy)
f.create_dataset('ref_z', data=ref_z)
f.create_dataset('bbsize1', data=bbsize)
f.close()


