__author__ = 'QiYE'

import h5py

dataset_path_prefix='/home/icvl/Qi/Prj_CNN_Hier_NewData/'
setname='mega'
source_name='_norm_hand_uvd_rootmid_half'
dataset='train'

src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
path = '%smega_%s%s.h5'%(src_path,dataset,source_name)


f = h5py.File(path,'r')

r0 = f['r0'][...]
print 'original train samples',r0.shape[0]
r1 = f['r1'][...]
r2= f['r2'][...]
uvd= f['uvd_jnt_gt_norm'][...]

f.close()
idx1 = range(0,r0.shape[0],4)

f= h5py.File('%smega_%s_norm_hand_uvd_rootmid_8.h5'%(src_path,dataset),'w')
f.create_dataset('r0',data=r0[idx1])
f.create_dataset('r1',data=r1[idx1])
f.create_dataset('r2',data=r2[idx1])
f.create_dataset('uvd_jnt_gt_norm',data=uvd[idx1])
f.close()

idx2 = range(0,r0.shape[0],8)

f= h5py.File('%smega_%s_norm_hand_uvd_rootmid_16.h5'%(src_path,dataset),'w')
f.create_dataset('r0',data=r0[idx2])
f.create_dataset('r1',data=r1[idx2])
f.create_dataset('r2',data=r2[idx2])
f.create_dataset('uvd_jnt_gt_norm',data=uvd[idx2])
f.close()



