__author__ = 'icvl'
import h5py
from src.utils import convert
import numpy


setname='mega'
dataset_path_prefix='/home/icvl/Qi/Prj_CNN_Hier_NewData/'
src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
path = '%s/train_norm_hand_uvd_rootmid.h5'%src_path


f = h5py.File(path,'r')#be carefull to run the code. wil change the source file. if run chnage 'r' to 'r+'
xyz_jnt_gt = f['xyz_jnt_gt'][...]
uvd = f['uvd_jnt_gt'][...]
ref_z = f['ref_z'][...]
hand_center_uvd = f['hand_center_uvd'][...]
mean_z = hand_center_uvd[:,2]

xyz_jnt_gt[:,:,2] = xyz_jnt_gt[:,:,2] - ref_z+mean_z.reshape(mean_z.shape[0],1)
data = f['xyz_jnt_gt']
data[...]=xyz_jnt_gt

f.close()


f = h5py.File(path,'r')
xyz_jnt_gt = f['xyz_jnt_gt'][...]
uvd = f['uvd_jnt_gt'][...]
bbsize = f['bbsize'][...]
ref_z = f['ref_z'][...]
hand_center_uvd = f['hand_center_uvd'][...]
resxy = f['resxy'][...]

f.close()

# mean_z = hand_center_uvd[:,2]
xyz = convert.uvd2xyz(uvd=uvd)
# xyz_jnt_gt[:,:,2] = xyz_jnt_gt[:,:,2] - ref_z+mean_z.reshape(mean_z.shape[0],1)
err =numpy.sqrt(numpy.sum((xyz-xyz_jnt_gt)**2,axis=-1))
print 'mean err ', numpy.mean(numpy.mean(err))


