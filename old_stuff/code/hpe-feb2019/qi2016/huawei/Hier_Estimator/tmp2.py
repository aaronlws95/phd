
import h5py
import numpy
batch_size=128
lr=0.0001
cur_finger=1
cur_jnt_idx=[cur_finger*4+1+1]
jnt_in_prev_layer=[cur_finger+1]
save_dir='/media/Data/Qi/data/hier'
version = 'pip_s0_finger%d_nojiter_ker48_lr%f'%(cur_finger,lr)
new_version = 'pip_s0_finger%d_smalljiter_ker48_lr%f'%(cur_finger,lr)

import os
os.rename('%s/history_%s_epoch.npy'%(save_dir,version), '%s/history_%s_epoch.npy'%(save_dir,new_version))
os.rename('%s/weight_%s'%(save_dir,version), '%s/weight_%s'%(save_dir,new_version))
os.rename('%s/%s.json'%(save_dir,version), '%s/%s.json'%(save_dir,new_version))


f = h5py.File('%s/source/test_crop_norm_vassi.h5'%save_dir, 'w')
new_file_names= f['new_file_names'][...]
f.close()



# f = h5py.File('%s/source/test_crop_norm_v1.h5'%save_dir, 'r+')
# train_y= f['uvd_norm_gt'][...].reshape(-1,63)
# print(train_y.shape)
# tmp = numpy.sum(numpy.abs(train_y)<0.5,axis=-1)
# loc=numpy.where(tmp==63)
# valid_idx = loc[0]
# numimg=int(valid_idx.shape[0]/batch_size)*batch_size
#
# print(valid_idx.shape)
# del f['valid_idx']
# f.create_dataset(name='valid_idx',data=valid_idx[:numimg])
# f.close()
# print('valid_idx added')
#
# f = h5py.File('%s/source/train_crop_norm_v1.h5'%save_dir, 'r+')
# train_y= f['uvd_norm_gt'][...].reshape(-1,63)
# print(train_y.shape)
# tmp = numpy.sum(numpy.abs(train_y)<0.5,axis=-1)
# loc=numpy.where(tmp==63)
# valid_idx = loc[0]
# numimg=int(valid_idx.shape[0]/batch_size)*batch_size
#
# print(valid_idx.shape)
# del f['valid_idx']
# f.create_dataset(name='valid_idx',data=valid_idx[:numimg])
# f.close()
# print('valid_idx added')
#
# f = h5py.File('%s/source/train_crop_norm_v1.h5'%save_dir, 'r')
# valid=f['valid_idx'][...]
# uvd_norm_gt= f['uvd_norm_gt'][...][valid]
# img0=f['img0'][...][valid]
# img1= f['img1'][...][valid]
# img2=f['img2'][...][valid]
# new_file_names= f['new_file_names'][...][valid]
# uvd_hand_centre=f['uvd_hand_centre'][...][valid]
# uvd_gt= f['uvd_gt'][...][valid]
# xyz_gt=f['xyz_gt'][...][valid]
# bbsize= f['bbsize'][...]
# f.close()
#
# numimg=valid.shape[0]
# idx1=list(range(0,403579,1))+list(range(448847,numimg,1))
# idx2=range(403579,448847,1)#the frames belonging to  vassilieos
#
# f = h5py.File('%s/source/train_crop_norm_vassi.h5'%save_dir, 'w')
# new_file_names=numpy.array(new_file_names,dtype=object)
# dt = h5py.special_dtype(vlen=str)
# f.create_dataset('new_file_names', data=new_file_names[idx1],dtype=dt)
# f.create_dataset('img0', data=img0[idx1])
# f.create_dataset('img1', data=img1[idx1])
# f.create_dataset('img2', data=img2[idx1])
# f.create_dataset('xyz_gt', data=xyz_gt[idx1])
# f.create_dataset('uvd_gt', data=uvd_gt[idx1])
# f.create_dataset('uvd_hand_centre', data=uvd_hand_centre[idx1])
# f.create_dataset('uvd_norm_gt', data=uvd_norm_gt[idx1])
# f.create_dataset('bbsize', data=bbsize)
# f.close()
#
# f = h5py.File('%s/source/test_crop_norm_vassi.h5'%save_dir, 'w')
# new_file_names=numpy.array(new_file_names,dtype=object)
# dt = h5py.special_dtype(vlen=str)
# f.create_dataset('new_file_names', data=new_file_names[idx2],dtype=dt)
# f.create_dataset('img0', data=img0[idx2])
# f.create_dataset('img1', data=img1[idx2])
# f.create_dataset('img2', data=img2[idx2])
# f.create_dataset('xyz_gt', data=xyz_gt[idx2])
# f.create_dataset('uvd_gt', data=uvd_gt[idx2])
# f.create_dataset('uvd_hand_centre', data=uvd_hand_centre[idx2])
# f.create_dataset('uvd_norm_gt', data=uvd_norm_gt[idx2])
# f.create_dataset('bbsize', data=bbsize)
# f.close()