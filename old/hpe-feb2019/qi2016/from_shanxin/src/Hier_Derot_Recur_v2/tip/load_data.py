import h5py
import numpy
import matplotlib.pyplot as plt
from src.utils.crop_patch_norm_offset import crop_patch,norm_offset_uvd,crop_patch_enlarge
from sklearn.utils import shuffle
def show_patch_offset_jnt(r0_patch,r1_patch,r2_patch,offset,r0,uvd,patch_size,patch_pad_width):
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
        # plt.scatter(numpy.mean(uvd[i,0])*96,numpy.mean(uvd[i,1])*96,c='r')
        # plt.scatter(uvd[i,0]*96,uvd[i,1]*96,c='g')
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

def load_data_multi_tip_uvd_normalized(batch_size,path,pre_jnt_path,jnt_idx,is_shuffle,patch_size=24, patch_pad_width=4,offset_depth_range=0.8,hand_width=96,hand_pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]
    f.close()
    prev_jnt_uvd_pred = numpy.load(pre_jnt_path)
    cur_uvd=numpy.squeeze(gr_uvd_derot[:,jnt_idx,:])
    # for i in numpy.random.randint(0,100,10):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(prev_jnt_uvd_pred[i,0]*96,prev_jnt_uvd_pred[i,1]*96)
    #     plt.scatter(cur_uvd[i,0]*96,cur_uvd[i,1]*96,c='r')
    #     plt.show()
    p0,p1,p2 = crop_patch(prev_jnt_uvd_pred,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset =(cur_uvd-  prev_jnt_uvd_pred)*10


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