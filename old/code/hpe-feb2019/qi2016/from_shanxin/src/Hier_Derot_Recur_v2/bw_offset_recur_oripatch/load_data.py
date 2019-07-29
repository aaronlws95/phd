__author__ = 'QiYE'
import h5py
import numpy
import matplotlib.pyplot as plt
from src.utils.crop_patch_norm_offset import crop_patch,norm_offset_uvd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from src.utils.normalize import norm_01


def show_patch(r0_patch,r1_patch,r2_patch,r0,gr_uvd_derot,pred_uvd_derot):
    num = 5
    index = numpy.random.randint(0,r0_patch.shape[0],num)
    for k in xrange(num):
        i = index[k]
        fig = plt.figure()
        ax= fig.add_subplot(221)
        ax.imshow(r0_patch[i],'gray')
        ax= fig.add_subplot(223)
        ax.imshow(r1_patch[i],'gray')
        ax= fig.add_subplot(224)
        ax.imshow(r2_patch[i],'gray')
        ax= fig.add_subplot(222)
        ax.imshow(r0[i],'gray')
        plt.scatter(gr_uvd_derot[i,0]*96,gr_uvd_derot[i,1]*96,c='r')
        plt.scatter(pred_uvd_derot[i,0]*96,pred_uvd_derot[i,1]*96,c='g')
        plt.title('%d'%i)
        plt.show()

def show_patch_offset(r0_patch,r1_patch,r2_patch,offset,patch_size,patch_pad_width):
    num=5
    index = numpy.arange(1,1+num,1)
    # index = numpy.random.randint(0,r0_patch.shape[0],10)
    for k in xrange(num):
        i = index[k]
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



def load_data_r012_ego_offset(batch_size,path,is_shuffle,jnt_idx,patch_size=44,patch_pad_width=4,hand_width=96,hand_pad_width=0):
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


def load_data_multi(batch_size,path,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')

    r0 = numpy.squeeze(f['patch00'][...][jnt_idx])
    r1 =  numpy.squeeze(f['patch10'][...][jnt_idx])
    r2=  numpy.squeeze(f['patch20'][...][jnt_idx])
    pred_uvd_derot = f['pred_uvd_derot'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]

    f.close()

    offset=numpy.squeeze((gr_uvd_derot.reshape(gr_uvd_derot.shape[0],6,3)-pred_uvd_derot.reshape(gr_uvd_derot.shape[0],6,3))[:,jnt_idx,:])*10
    # show_hist(offset[:,0])
    # show_hist(offset[:,1])
    # show_hist(offset[:,2])
    # print offset[numpy.where(offset>1)]
    # print offset[numpy.where(offset<-1)]
    # offset[numpy.where(offset>1)]=0
    # offset[numpy.where(offset<-1)]=0


    r0=norm_01(r0)
    r1=norm_01(r1)
    r2=norm_01(r2)
    if is_shuffle:
        r0,r1,r2,offset = shuffle(r0,r1,r2,offset,random_state=0)

        num = batch_size - r0.shape[0]%batch_size
        return numpy.concatenate([r0,r0[0:num]],axis=0),numpy.concatenate([r1,r1[0:num]],axis=0),numpy.concatenate([r2,r2[0:num]],axis=0),numpy.concatenate([offset,offset[0:num]],axis=0)
    else:
        return r0,r1,r2,offset
