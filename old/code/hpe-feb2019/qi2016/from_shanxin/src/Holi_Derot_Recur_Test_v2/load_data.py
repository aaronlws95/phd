import h5py
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy
from src.utils.normalize import norm_01
from src.utils.crop_patch_norm_offset import crop_patch,norm_offset_uvd

def load_data_iter0(batch_size,path,jnt_idx,is_shuffle):

    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    uvd = f['joint_label_uvd'][...][:,jnt_idx,:]
    f.close()

    print 'nsamples', r0.shape[0]

    if is_shuffle:
        r0,r1,r2,uvd = shuffle(r0,r1,r2,uvd,random_state=0)

        num = batch_size - r0.shape[0]%batch_size


        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])
    else:
        return r0.reshape(r0.shape[0], 1, r0.shape[1],r0.shape[2]),r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),\
               r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),uvd.reshape(uvd.shape[0], uvd.shape[1]*uvd.shape[2])


def load_data_iter1(batch_size,path,is_shuffle,jnt_idx,patch_size=44,patch_pad_width=4,hand_width=96,hand_pad_width=0):
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



def load_data_finger_jnt_iter0(batch_size,path,jnt_idx,is_shuffle,patch_size=24,patch_pad_width=4,hand_width=96,hand_pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''
    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]
    pred_whl_uvd_derot = f['pred_uvd_derot'][...]
    f.close()


    print 'index in whole hand',jnt_idx
    prev_iter_pred = numpy.squeeze(pred_whl_uvd_derot[:,jnt_idx,:])
    cur_uvd = numpy.squeeze(gr_uvd_derot[:,jnt_idx,:])
    #
    # for i in xrange(0,10,1):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(cur_uvd[i,0]*96,cur_uvd[i,1]*96,c='g')
    #     plt.scatter(prev_iter_pred[i,0]*96,prev_iter_pred[i,1]*96,c='r')
    #     plt.show()

    p0,p1,p2 = crop_patch(prev_iter_pred,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset=(cur_uvd-prev_iter_pred)*10

    print numpy.max(offset[:,0]), numpy.min(offset[:,0])
    print numpy.max(offset[:,1]), numpy.min(offset[:,1])
    print numpy.max(offset[:,2]), numpy.min(offset[:,2])

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

