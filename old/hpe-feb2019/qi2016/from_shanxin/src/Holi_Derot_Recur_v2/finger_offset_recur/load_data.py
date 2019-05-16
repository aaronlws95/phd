__author__ = 'QiYE'
import h5py
import numpy
import matplotlib.pyplot as plt
from src.utils.crop_patch_norm_offset import crop_patch,norm_offset_uvd,crop_patch_enlarge
from sklearn.utils import shuffle
from src.utils.show_statistics import show_hist
def show_patch_offset_jnt(r0_patch,r1_patch,r2_patch,offset,r0,uvd,patch_size,patch_pad_width):
    # index = numpy.arange(0,10,1)
    num=10
    index = numpy.random.randint(0,r0_patch.shape[0],num)
    for k in xrange(num):
        i=index[k]
        fig = plt.figure()
        ax= fig.add_subplot(221)
        ax.imshow(r0_patch[i],'gray')
        plt.scatter(patch_size/2+patch_pad_width,patch_size/2+patch_pad_width,c='r')
        plt.scatter(offset[i,0]*patch_size+patch_pad_width,offset[i,1]*patch_size+patch_pad_width,c='g')
        ax= fig.add_subplot(223)
        ax.imshow(r1_patch[i],'gray')
        ax= fig.add_subplot(224)
        ax.imshow(r2_patch[i],'gray')
        # ax= fig.add_subplot(222)
        # ax.imshow(r0[i],'gray')
        # plt.scatter(numpy.mean(uvd[i,:,0])*96,numpy.mean(uvd[i,:,1])*96,c='r')
        # plt.scatter(uvd[i,:,0]*96,uvd[i,:,1]*96,c='g')
        plt.title('%d'%i)
        plt.show()

def show_patch_ori_offset_jnt(r0_patch,r1_patch,r2_patch,offset,r0,uvd,patch_size,hand_width,patch_pad_width):
    num=20
    index = numpy.random.randint(0,r0_patch.shape[0],num)
    for k in xrange(num):
        i=index[k]
        fig = plt.figure()
        ax= fig.add_subplot(221)
        ax.imshow(r0_patch[i],'gray')
        plt.scatter(patch_size/2+patch_pad_width,patch_size/2+patch_pad_width,c='r')
        plt.scatter(offset[i,0]*hand_width+patch_pad_width+patch_size/2,offset[i,1]*hand_width+patch_size/2+patch_pad_width,c='g')
        ax= fig.add_subplot(223)
        ax.imshow(r1_patch[i],'gray')
        ax= fig.add_subplot(224)
        ax.imshow(r2_patch[i],'gray')
        # ax= fig.add_subplot(222)
        # ax.imshow(r0[i],'gray')
        # plt.scatter(numpy.mean(uvd[i,:,0])*96,numpy.mean(uvd[i,:,1])*96,c='r')
        # plt.scatter(uvd[i,:,0]*96,uvd[i,:,1]*96,c='g')
        plt.title('%d'%i)
        plt.show()

def load_data_multi_offset(batch_size,path,jnt_idx,is_shuffle,patch_size=24,patch_pad_width=4,hand_width=96,hand_pad_width=0):
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


def load_data_multi_offset_all(path,jnt_idx,is_shuffle,patch_size=24,patch_pad_width=4,hand_width=96,hand_pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''
    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]
    pred_whl_uvd_derot = f['pred_whl_uvd_derot'][...]
    f.close()


    print 'index in whole hand',jnt_idx[0]
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


    # show_patch_offset_jnt(p0,p1,p2,offset,r0,joint_label_uvd[:,jnt_idx,:],patch_size=patch_size,patch_pad_width=4)
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
    if is_shuffle == True:
        p0, p1, p2 ,offset= shuffle(p0, p1, p2,offset)
    return p0, p1, p2, offset