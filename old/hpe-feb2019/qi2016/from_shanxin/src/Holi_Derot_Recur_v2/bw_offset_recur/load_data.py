__author__ = 'QiYE'
import h5py
import numpy
from src.utils.crop_patch_norm_offset import crop_patch
from sklearn.utils import shuffle

def load_data_r012_ego_offset(batch_size,path,is_shuffle,jnt_idx,patch_size=44,patch_pad_width=4,hand_width=96,hand_pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]
    pred_uvd_derot = f['pred_uvd_derot'][...]
    f.close()
    prev_uvd_derot = numpy.squeeze(pred_uvd_derot[:,jnt_idx,:])
    cur_uvd_derot = numpy.squeeze(gr_uvd_derot[:,jnt_idx,:])


    p0,p1,p2 = crop_patch(prev_uvd_derot,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset =(cur_uvd_derot-prev_uvd_derot)*10



    print 'u beyond [0,1]',numpy.where(offset[:,0]<0)[0].shape[0]+numpy.where(offset[:,0]>1)[0].shape[0]
    print 'v beyond [0,1]',numpy.where(offset[:,1]<0)[0].shape[0]+numpy.where(offset[:,1]>1)[0].shape[0]
    print 'd beyond [0,1]',numpy.where(offset[:,2]<0)[0].shape[0]+numpy.where(offset[:,2]>1)[0].shape[0]

    if is_shuffle:
        p0,p1,p2,offset = shuffle(p0,p1,p2,offset,random_state=0)

        num = batch_size - r0.shape[0]%batch_size


        return numpy.concatenate([p0,p0[0:num]],axis=0).reshape(p0.shape[0]+num, 1, p0.shape[1],p0.shape[2]), \
               numpy.concatenate([p1,p1[0:num]],axis=0).reshape(p1.shape[0]+num, 1, p1.shape[1],p1.shape[2]),\
               numpy.concatenate([p2,p2[0:num]],axis=0).reshape(p2.shape[0]+num, 1, p2.shape[1],p2.shape[2]),\
               numpy.concatenate([offset,offset[0:num]],axis=0)
    else:
        return p0.reshape(p0.shape[0], 1, p0.shape[1],p0.shape[2]),p1.reshape(p1.shape[0], 1, p1.shape[1],p1.shape[2]),\
               p2.reshape(p2.shape[0], 1, p2.shape[1],p2.shape[2]),offset
