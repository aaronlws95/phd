import h5py
from sklearn.utils import shuffle
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
    pred_bw_uvd_derot = f['pred_bw_uvd_derot'][...]
    f.close()
    bw_idx = (jnt_idx[0]+3)/4
    print 'bw_idx',bw_idx

    prev_uvd_derot = numpy.squeeze(pred_bw_uvd_derot[:,bw_idx,:])
    cur_uvd_derot = numpy.squeeze(gr_uvd_derot[:,jnt_idx,:])
    # for i in xrange(10,20,1):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(cur_uvd_derot[i,0]*96,cur_uvd_derot[i,1]*96,c='g')
    #     plt.scatter(prev_uvd_derot[i,0]*96,prev_uvd_derot[i,1]*96,c='r')
    #     plt.show()


    p0,p1,p2 = crop_patch(prev_uvd_derot,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset =(cur_uvd_derot-prev_uvd_derot)*10
    # offset = norm_offset_uvd(cur_uvd=cur_uvd_derot,prev_uvd=prev_uvd_derot,offset_depth_range=0.4,patch_size=patch_size,hand_width=hand_width)


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


def load_data_finger_jnt(path,prev_jnt_uvd_pred,jnt_idx,is_shuffle,patch_size=24,patch_pad_width=4,hand_width=96,hand_pad_width=0,offset_depth_range=0.8):
    '''     mid, tip and top jnt has the same load procedure
    '''
    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['gr_uvd_derot'][...]
    f.close()

    cur_uvd=numpy.squeeze(joint_label_uvd[:,jnt_idx,:])

    p0,p1,p2 = crop_patch(prev_jnt_uvd_pred,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset = norm_offset_uvd(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd_pred,offset_depth_range=offset_depth_range,
                                                                patch_size=patch_size,hand_width=hand_width)
    print 'u beyond [0,1]',numpy.where(offset[:,0]<0)[0].shape[0]+numpy.where(offset[:,0]>1)[0].shape[0]
    print 'v beyond [0,1]',numpy.where(offset[:,1]<0)[0].shape[0]+numpy.where(offset[:,1]>1)[0].shape[0]
    print 'd beyond [0,1]',numpy.where(offset[:,2]<0)[0].shape[0]+numpy.where(offset[:,2]>1)[0].shape[0]
    # show_patch_offset_jnt(p0,p1,p2,offset,r0,joint_label_uvd[:,jnt_idx,:],patch_size=patch_size,patch_pad_width=4)
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
    if is_shuffle == True:
        p0, p1, p2 ,offset= shuffle(p0, p1, p2,offset)

    # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
    return p0, p1, p2, offset





