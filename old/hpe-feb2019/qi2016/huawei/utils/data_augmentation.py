__author__ = 'QiYE'
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
import numpy
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

import math


def augment_data_3d_mega_rot_scale(r0,r1,r2,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_r1=r1[:,:,:,0].copy()
    new_r2=r2[:,:,:,0].copy()

    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.05,size=num_frame)

    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame*0.5)):
        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((48,48),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96
        M = cv2.getRotationMatrix2D((24,24),rot[i],scale_factor[i])
        new_r1[i] = cv2.warpAffine(new_r1[i],M,(48,48),borderValue=1)
        M = cv2.getRotationMatrix2D((12,12),rot[i],scale_factor[i])
        new_r2[i] = cv2.warpAffine(new_r2[i],M,(24,24),borderValue=1)

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),numpy.expand_dims(new_r1,axis=-1),\
           numpy.expand_dims(new_r2,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])



def augment_data_3d_mega_trans_rot_scale(r0,r1,r2,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_r1=r1[:,:,:,0].copy()
    new_r2=r2[:,:,:,0].copy()

    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.05,size=num_frame)
    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.05,size=num_frame)



    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame*0.5)):
        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        """depth translation"""
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0

        new_gr_uvd[i,:,2]+=center_z[i]


        loc = numpy.where(new_r1[i]<1.0)
        new_r1[i][loc]+=center_z[i]
        loc = numpy.where(new_r1[i]>1.0)
        new_r1[i][loc]=1.0


        loc = numpy.where(new_r2[i]<1.0)
        new_r2[i][loc]+=center_z[i]
        loc = numpy.where(new_r2[i]>1.0)
        new_r2[i][loc]=1.0


        """2d translation, rotation and scale"""

        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        M = cv2.getRotationMatrix2D((center_x[i]/2,center_y[i]/2),rot[i],scale_factor[i])
        new_r1[i] = cv2.warpAffine(new_r1[i],M,(48,48),borderValue=1)
        M = cv2.getRotationMatrix2D((center_x[i]/4,center_y[i]/4),rot[i],scale_factor[i])
        new_r2[i] = cv2.warpAffine(new_r2[i],M,(24,24),borderValue=1)

        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        # fig = plt.figure()
        # ax =fig.add_subplot(221)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r')
        # ax =fig.add_subplot(222)
        # ax.imshow(new_r1[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*48+24,new_gr_uvd[i,:,1]*48+24,c='r')
        # ax =fig.add_subplot(223)
        # ax.imshow(new_r2[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*24+12,new_gr_uvd[i,:,1]*24+12,c='r')
        #
        # tmp=gr_uvd[i].reshape(6,3)
        # # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(224)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),numpy.expand_dims(new_r1,axis=-1),numpy.expand_dims(new_r2,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])


def crop_patch_onejnt(r0,pred_uv,r0_patch_size):
    border=r0_patch_size
    jnt_uv=pred_uv.copy()

    jnt_uv[numpy.where(pred_uv>r0.shape[0]+r0_patch_size/2-1)]=r0.shape[0]+r0_patch_size/2-1
    jnt_uv[numpy.where(pred_uv<-r0_patch_size/2)]=-r0_patch_size/2
    jnt_uv=numpy.array(numpy.round(jnt_uv),dtype='int16')
    # print(pred_uv,jnt_uv)
    r0_patch_half_size = int(r0_patch_size/2)

    r0_tmp = numpy.lib.pad(r0, ((border,border),(border,border)), 'constant',constant_values=1)

    vmin = jnt_uv[1]-r0_patch_half_size+border
    vmax = jnt_uv[1]+r0_patch_half_size+border
    umin= jnt_uv[0]-r0_patch_half_size+border
    umax =jnt_uv[0]+r0_patch_half_size+border
    p1 = r0_tmp[vmin:vmax,umin:umax]

    return p1




def get_hand_part_for_palm_old(r0,gr_uvd,pred_uvd,jnt,if_aug=True):
    new_r0=r0[:,:,:,0].copy()
    num_frame=gr_uvd.shape[0]
    rot_angle = math.get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])

    crop0=numpy.empty((num_frame,48,48,1),dtype='float32')
    crop1 = numpy.empty((num_frame,24,24,1),dtype='float32')
    target = numpy.empty((num_frame,3),dtype='float32')
    if if_aug:
        aug_frame = numpy.random.uniform(0,1,num_frame)
        aug_frame = numpy.where(aug_frame>0.5,1,0)
    else:
        aug_frame=numpy.zeros((num_frame,),dtype='uint8')

    for i in range(0,gr_uvd.shape[0],1):
        cur_jnt_gt_uvd = gr_uvd[i,jnt,:]
        cur_jnt_pred_uvd = pred_uvd[i,jnt,:]
        target[i,2]=cur_jnt_gt_uvd[2]-cur_jnt_pred_uvd[2]

        if aug_frame[i]:
            cur_jnt_pred_uvd+= numpy.random.normal(loc=0,scale=0.05,size=3)
            rot=numpy.random.normal(loc=0,scale=15,size=1)
            scale=numpy.random.normal(loc=1.5,scale=0.1,size=1)
        else:
            scale=1.5
            rot=0

        M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],scale)
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)


        tmp_gt_uv = numpy.dot(M,numpy.array([cur_jnt_gt_uvd[0]*96+48,cur_jnt_gt_uvd[1]*96+48,1]))
        tmp_pred_uv= numpy.dot(M,numpy.array([cur_jnt_pred_uvd[0]*96+48,cur_jnt_pred_uvd[1]*96+48,1]))

        crop0[i,:,:,0]=crop_patch_onejnt(r0=new_r0[i],pred_uv=tmp_pred_uv,r0_patch_size=48)
        crop1[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)
        target[i,0:2]=(tmp_gt_uv-tmp_pred_uv)/96.0

        # fig = plt.figure()
        # ax =fig.add_subplot(221)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(tmp_gt_uv[0],tmp_gt_uv[1],c='r')
        # ax.scatter(tmp_pred_uv[0],tmp_pred_uv[1],c='r')
        #
        # ax =fig.add_subplot(222)
        # ax.imshow(crop1[i,:,:,0],'gray')
        #
        # ax =fig.add_subplot(223)
        # ax.imshow(crop0[i,:,:,0],'gray')
        # ax.scatter(target[i,0]*48+24,target[i,1]*48+24,c='r')
        #
        # ax =fig.add_subplot(224)
        # ax.imshow(r0[i,:,:,0],'gray')
        # # plt.scatter(pred_uvd[i,:,0]*96+48,pred_uvd[i,:,1]*96+48,c='b')
        # # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()

    return crop0,crop1,target*10

def get_hand_part_for_palm(r0,gr_uvd,pred_uvd,jnt,if_aug=True):
    # new_r0=r0[:,:,:,0].copy()
    num_frame=gr_uvd.shape[0]
    rot_angle = math.get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])

    crop0=numpy.empty((num_frame,48,48,1),dtype='float32')
    crop1 = numpy.empty((num_frame,24,24,1),dtype='float32')
    target = numpy.empty((num_frame,3),dtype='float32')

    if if_aug:
        aug_frame = numpy.random.uniform(0,1,num_frame)
        aug_frame = numpy.where(aug_frame>0.5,1,0)
    else:
        aug_frame=numpy.zeros((num_frame,),dtype='uint8')

    for i in range(0,gr_uvd.shape[0],1):
    # for i in numpy.random.randint(0,gr_uvd.shape[0],30):
        cur_jnt_gt_uvd = gr_uvd[i,jnt,:]
        cur_jnt_pred_uvd = pred_uvd[i,jnt,:]
        target[i,2]=cur_jnt_gt_uvd[2]-cur_jnt_pred_uvd[2]

        if aug_frame[i]:
            cur_jnt_pred_uvd+= numpy.random.normal(loc=0,scale=0.005,size=3)
            rot=numpy.random.normal(loc=0,scale=20,size=1)
            scale=numpy.random.normal(loc=1.8,scale=0.1,size=1)
            # print(rot,scale)
        else:
            scale=1.8
            rot=0

        offset_uvd = cur_jnt_gt_uvd-cur_jnt_pred_uvd
        tx=-cur_jnt_pred_uvd[0]*96#cols
        ty=-cur_jnt_pred_uvd[1]*96#rows
        M = numpy.float32([[1,0,tx],[0,1,ty]])
        dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # dst1 = cv2.warpAffine(r0[i],M,(48,48),borderValue=0)

        M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],scale)
        dst= cv2.warpAffine(dst,M,(96,96),borderValue=1)
        crop0[i,:,:,0]=dst[24:72,24:72]


        offset_uvd[0:2] = (numpy.dot(M,numpy.array([offset_uvd[0]*96+48,offset_uvd[1]*96+48,1]))-48)/96
        target[i]=offset_uvd
        crop1[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)
        # target[i,0:2]=(tmp_gt_uv-tmp_pred_uv)/96.0

        # fig = plt.figure()
        # ax =fig.add_subplot(221)
        # ax.imshow(crop0[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[0]*48+24,offset_uvd[1]*48+24,c='b')
        # ax.scatter(24,24,c='r')
        # # ax.scatter(tmp_pred_uv[0],tmp_pred_uv[1],c='r')
        # #
        # ax =fig.add_subplot(222)
        # ax.imshow(dst,'gray')
        # ax.scatter(offset_uvd[0]*96+48,offset_uvd[1]*96+48,c='b')
        # ax.scatter(48,48,c='r')
        #
        # ax =fig.add_subplot(223)
        # ax.imshow(crop1[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[0]*24+12,offset_uvd[1]*24+12,c='b')
        # ax.scatter(12,12,c='r')
        #
        # ax =fig.add_subplot(224)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(pred_uvd[i,:,0]*96+48,pred_uvd[i,:,1]*96+48,c='b')
        # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()

    return crop0,crop1,target*10

def get_hand_part_for_palm_holi(r0,gr_uvd,pred_uvd,uvd_hand_center,if_aug=True):
    new_r0=r0.copy()
    num_frame=gr_uvd.shape[0]
    rot_angle = math.get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])

    crop0=numpy.empty((num_frame,48,48,1),dtype='float32')
    crop1 = numpy.empty((num_frame,24,24,1),dtype='float32')
    target = numpy.empty((num_frame,6,3),dtype='float32')

    if if_aug:
        # aug_frame=numpy.ones((num_frame,),dtype='uint8')
        aug_frame = numpy.random.uniform(0,1,num_frame)
        aug_frame = numpy.where(aug_frame>0.5,1,0)
    else:
        aug_frame=numpy.zeros((num_frame,),dtype='uint8')
    for i in range(gr_uvd.shape[0]):
        cur_gt_uvd = gr_uvd[i,:,:]
        cur_pred_uvd = pred_uvd[i,:,:]
        mean_pred_uvd = numpy.mean(cur_pred_uvd,axis=0)

        if aug_frame[i]:
            mean_pred_uvd+= numpy.random.normal(loc=0,scale=0.05,size=3)
            rot=numpy.random.normal(loc=0,scale=30,size=1)
            scale=numpy.random.normal(loc=1.2,scale=0.05,size=1)
            print(rot,scale)
        else:
            scale=1.2
            rot=0

        #mean_pred_uvd is the new normcenter

        offset_uvd = cur_gt_uvd-numpy.expand_dims(mean_pred_uvd,axis=0)

        "depth translation"
        loc = numpy.where(new_r0[i,:,:,0]<1.0)
        new_r0[i,:,:,0][loc]-=mean_pred_uvd[2]
        loc = numpy.where(new_r0[i,:,:,0]>1.0)
        new_r0[i,:,:,0][loc]=1.0

        "2D translation"
        tx=-mean_pred_uvd[0]*96#cols
        ty=-mean_pred_uvd[1]*96#rows

        "scale change together with inplane rotation change"
        #size of the norm image is in proportion of the center depth value
        # pred_scale=1/uvd_hand_center[2]
        # now_scale=1/mean_pred_uvd[2]
        scale_ratio = uvd_hand_center[i,2]/(mean_pred_uvd[2]*300+uvd_hand_center[i,2])
        # print(uvd_hand_center[i,2],mean_pred_uvd[2],scale_ratio)

        M = numpy.float32([[1,0,tx],[0,1,ty]])
        dst = cv2.warpAffine(new_r0[i,:,:,0],M,(96,96),borderValue=1)
        # dst1 = cv2.warpAffine(r0[i],M,(48,48),borderValue=0)

        M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],scale*scale_ratio)
        dst= cv2.warpAffine(dst,M,(96,96),borderValue=1)


        for j in range(offset_uvd.shape[0]):
            offset_uvd[j,0:2] = (numpy.dot(M,numpy.array([offset_uvd[j,0]*96+48,offset_uvd[j,1]*96+48,1]))-48)/96

        target[i]=offset_uvd

        crop0[i,:,:,0]=dst[24:72,24:72]
        crop1[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)
        # target[i,0:2]=(tmp_gt_uv-tmp_pred_uv)/96.0

        # fig = plt.figure()
        # ax =fig.add_subplot(221)
        # ax.imshow(crop0[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[:,0]*96+24,offset_uvd[:,1]*96+24,c='b')
        # ax.scatter(24,24,c='r')
        # # ax.scatter(tmp_pred_uv[0],tmp_pred_uv[1],c='r')
        # #
        # ax =fig.add_subplot(222)
        # ax.imshow(dst,'gray')
        # ax.scatter(offset_uvd[:,0]*96+48,offset_uvd[:,1]*96+48,c='b')
        # ax.scatter(48,48,c='r')
        #
        # ax =fig.add_subplot(223)
        # ax.imshow(crop1[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[:,0]*48+12,offset_uvd[:,1]*48+12,c='b')
        # ax.scatter(12,12,c='r')
        #
        # ax =fig.add_subplot(224)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(pred_uvd[i,:,0]*96+48,pred_uvd[i,:,1]*96+48,c='b')
        # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()
    return crop0,crop1,target.reshape(target.shape[0],-1)


def get_img_for_palm_holi(r0,gr_uvd,pred_uvd,uvd_hand_center,if_aug=True):
    new_r0=r0.copy()
    num_frame=gr_uvd.shape[0]
    rot_angle = math.get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])

    crop1=numpy.empty((num_frame,48,48,1),dtype='float32')
    crop2 = numpy.empty((num_frame,24,24,1),dtype='float32')
    crop0 = numpy.empty((num_frame,96,96,1),dtype='float32')
    target = numpy.empty((num_frame,6,3),dtype='float32')

    if if_aug:
        # aug_frame=numpy.ones((num_frame,),dtype='uint8')
        aug_frame = numpy.random.uniform(0,1,num_frame)
        aug_frame = numpy.where(aug_frame>0.5,1,0)
    else:
        aug_frame=numpy.zeros((num_frame,),dtype='uint8')
    for i in range(gr_uvd.shape[0]):
        cur_gt_uvd = gr_uvd[i,:,:]
        cur_pred_uvd = pred_uvd[i,:,:]
        mean_pred_uvd = numpy.mean(cur_pred_uvd,axis=0)

        if aug_frame[i]:
            mean_pred_uvd+= numpy.random.normal(loc=0,scale=0.05,size=3)
            rot=numpy.random.normal(loc=0,scale=15,size=1)
        else:
            rot=0

        #mean_pred_uvd is the new normcenter
        offset_uvd = cur_gt_uvd-numpy.expand_dims(mean_pred_uvd,axis=0)
        "depth translation"
        loc = numpy.where(new_r0[i,:,:,0]<1.0)
        new_r0[i,:,:,0][loc]-=mean_pred_uvd[2]
        loc = numpy.where(new_r0[i,:,:,0]>1.0)
        new_r0[i,:,:,0][loc]=1.0

        "2D translation"
        tx=-mean_pred_uvd[0]*96#cols
        ty=-mean_pred_uvd[1]*96#rows

        "scale change together with inplane rotation change"
        #size of the norm image is in proportion of the center depth value
        scale_ratio = (mean_pred_uvd[2]*300+uvd_hand_center[i,2])/uvd_hand_center[i,2]

        M = numpy.float32([[1,0,tx],[0,1,ty]])
        dst = cv2.warpAffine(new_r0[i,:,:,0],M,(96,96),borderValue=1)
        # dst1 = cv2.warpAffine(r0[i],M,(48,48),borderValue=0)

        M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],1.8*scale_ratio)
        dst= cv2.warpAffine(dst,M,(96,96),borderValue=1)


        for j in range(offset_uvd.shape[0]):
            offset_uvd[j,0:2] = (numpy.dot(M,numpy.array([offset_uvd[j,0]*96+48,offset_uvd[j,1]*96+48,1]))-48)/96

        target[i]=offset_uvd

        crop0[i,:,:,0]=dst
        crop1[i,:,:,0] = resize(crop0[i,:,:,0], (48,48), order=3,preserve_range=True)
        crop2[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)

        # fig = plt.figure()
        # ax =fig.add_subplot(221)
        # ax.imshow(crop0[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[:,0]*96+48,offset_uvd[:,1]*96+48,c='b')
        # ax.scatter(48,48,c='r')
        # # ax.scatter(tmp_pred_uv[0],tmp_pred_uv[1],c='r')
        # ax =fig.add_subplot(222)
        # ax.imshow(crop1[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[:,0]*48+24,offset_uvd[:,1]*48+24,c='b')
        # ax.scatter(24,24,c='r')
        #
        # ax =fig.add_subplot(223)
        # ax.imshow(crop2[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[:,0]*24+12,offset_uvd[:,1]*24+12,c='b')
        # ax.scatter(12,12,c='r')
        #
        # ax =fig.add_subplot(224)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(pred_uvd[i,:,0]*96+48,pred_uvd[i,:,1]*96+48,c='b')
        # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()
    return crop0,crop1,crop2,target.reshape(target.shape[0],-1)

def get_crop_for_finger_part_s0(r0,gr_uvd,pred_uvd,jnt_uvd_in_prev_layer,if_aug=True,aug_trans=0.05, aug_rot=15,scale=1.8):
    new_r0=r0.copy()
    num_frame=gr_uvd.shape[0]
    rot_angle = math.get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])

    crop0=numpy.empty((num_frame,48,48,1),dtype='float32')
    crop1 = numpy.empty((num_frame,24,24,1),dtype='float32')
    target = numpy.empty_like(gr_uvd)

    if if_aug:
        # aug_frame=numpy.ones((num_frame,),dtype='uint8')
        aug_frame = numpy.random.uniform(0,1,num_frame)
        aug_frame = numpy.where(aug_frame>0.5,1,0)
    else:
        aug_frame=numpy.zeros((num_frame,),dtype='uint8')
    for i in range(gr_uvd.shape[0]):
        cur_gt_uvd = gr_uvd[i]
        cur_pred_uvd=jnt_uvd_in_prev_layer[i]
        # print(cur_pred_uvd.shape,cur_pred_uvd.shape)

        if aug_frame[i]:
            cur_pred_uvd+= numpy.random.normal(loc=0,scale=aug_trans,size=3)
            rot=numpy.random.normal(loc=0,scale=aug_rot,size=1)
        else:
            rot=0
        #mean_pred_uvd is the new normcenter
        offset_uvd = cur_gt_uvd-cur_pred_uvd
        # print(offset_uvd.shape)

        "2D translation"
        tx=-cur_pred_uvd[0,0]*96#cols
        ty=-cur_pred_uvd[0,1]*96#rows

        M = numpy.float32([[1,0,tx],[0,1,ty]])
        dst = cv2.warpAffine(new_r0[i,:,:,0],M,(96,96),borderValue=1)

        M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],scale=scale)
        dst= cv2.warpAffine(dst,M,(96,96),borderValue=1)
        for j in range(offset_uvd.shape[0]):
            offset_uvd[j,0:2] = (numpy.dot(M,numpy.array([offset_uvd[j,0]*96+48,offset_uvd[j,1]*96+48,1]))-48)/96

        target[i]=offset_uvd

        crop0[i,:,:,0]=dst[24:72,24:72]
        crop1[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)

        # fig = plt.figure()
        # ax =fig.add_subplot(221)
        # ax.imshow(crop0[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[0]*96+24,offset_uvd[1]*96+24,c='b')
        # ax.scatter(24,24,c='r')
        # # ax.scatter(tmp_pred_uv[0],tmp_pred_uv[1],c='r')
        # ax =fig.add_subplot(222)
        # ax.imshow(crop1[i,:,:,0],'gray')
        # ax.scatter(offset_uvd[0]*48+12,offset_uvd[1]*48+12,c='b')
        # ax.scatter(12,12,c='r')
        #
        # # ax =fig.add_subplot(223)
        # # ax.imshow(crop2[i,:,:,0],'gray')
        # # ax.scatter(offset_uvd[:,0]*24+12,offset_uvd[:,1]*24+12,c='b')
        # # ax.scatter(12,12,c='r')
        # #
        # ax =fig.add_subplot(224)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(pred_uvd[i,:,0]*96+48,pred_uvd[i,:,1]*96+48,c='r')
        # plt.scatter(gr_uvd[i,0]*96+48,gr_uvd[i,1]*96+48,c='b')
        # plt.show()
    return crop0,crop1,target.reshape(target.shape[0],-1)

# def get_crop_for_dtip_s0(r0,gr_uvd,pred_uvd,jnt_uvd_in_prev_layer,if_aug=True,scale=1.8):
#     new_r0=r0.copy()
#     num_frame=gr_uvd.shape[0]
#     rot_angle = math.get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])
#
#     crop0=numpy.empty((num_frame,48,48,1),dtype='float32')
#     crop1 = numpy.empty((num_frame,24,24,1),dtype='float32')
#     target = numpy.empty((num_frame,3),dtype='float32')
#
#     if if_aug:
#         # aug_frame=numpy.ones((num_frame,),dtype='uint8')
#         aug_frame = numpy.random.uniform(0,1,num_frame)
#         aug_frame = numpy.where(aug_frame>0.5,1,0)
#     else:
#         aug_frame=numpy.zeros((num_frame,),dtype='uint8')
#     for i in range(gr_uvd.shape[0]):
#         cur_gt_uvd = gr_uvd[i]
#         cur_pred_uvd=jnt_uvd_in_prev_layer[i]
#         # print(cur_pred_uvd.shape,cur_pred_uvd.shape)
#
#         if aug_frame[i]:
#             cur_pred_uvd+= numpy.random.normal(loc=0,scale=0.05,size=3)
#             rot=numpy.random.normal(loc=0,scale=15,size=1)
#         else:
#             rot=0
#         #mean_pred_uvd is the new normcenter
#         offset_uvd = cur_gt_uvd-cur_pred_uvd
#         # print(offset_uvd.shape)
#
#         "2D translation"
#         tx=-cur_pred_uvd[0]*96#cols
#         ty=-cur_pred_uvd[1]*96#rows
#         M = numpy.float32([[1,0,tx],[0,1,ty]])
#         dst = cv2.warpAffine(new_r0[i,:,:,0],M,(96,96),borderValue=1)
#         M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],scale=scale)
#         dst= cv2.warpAffine(dst,M,(96,96),borderValue=1)
#         for j in range(offset_uvd.shape[0]):
#             offset_uvd[j,0:2] = (numpy.dot(M,numpy.array([offset_uvd[j,0]*96+48,offset_uvd[j,1]*96+48,1]))-48)/96
#
#         target[i]=offset_uvd
#
#         crop0[i,:,:,0]=dst[24:72,24:72]
#         crop1[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)
#
#         # fig = plt.figure()
#         # ax =fig.add_subplot(221)
#         # ax.imshow(crop0[i,:,:,0],'gray')
#         # ax.scatter(offset_uvd[0]*96+24,offset_uvd[1]*96+24,c='b')
#         # ax.scatter(24,24,c='r')
#         # # ax.scatter(tmp_pred_uv[0],tmp_pred_uv[1],c='r')
#         # ax =fig.add_subplot(222)
#         # ax.imshow(crop1[i,:,:,0],'gray')
#         # ax.scatter(offset_uvd[0]*48+12,offset_uvd[1]*48+12,c='b')
#         # ax.scatter(12,12,c='r')
#         #
#         # # ax =fig.add_subplot(223)
#         # # ax.imshow(crop2[i,:,:,0],'gray')
#         # # ax.scatter(offset_uvd[:,0]*24+12,offset_uvd[:,1]*24+12,c='b')
#         # # ax.scatter(12,12,c='r')
#         # #
#         # ax =fig.add_subplot(224)
#         # ax.imshow(r0[i,:,:,0],'gray')
#         # plt.scatter(pred_uvd[i,:,0]*96+48,pred_uvd[i,:,1]*96+48,c='r')
#         # plt.scatter(gr_uvd[i,0]*96+48,gr_uvd[i,1]*96+48,c='b')
#         # plt.show()
#     return crop0,crop1,target
#



def get_derot_images(r0,gr_uvd,pred_uvd,uvd_hand_center,if_aug=True):
    new_r0=r0.copy()
    num_frame=gr_uvd.shape[0]
    rot_angle = math.get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])

    crop1=numpy.empty((num_frame,48,48,1),dtype='float32')
    crop2 = numpy.empty((num_frame,24,24,1),dtype='float32')
    crop0 = numpy.empty((num_frame,96,96,1),dtype='float32')
    target = numpy.empty((num_frame,6,3),dtype='float32')

    if if_aug:
        # aug_frame=numpy.ones((num_frame,),dtype='uint8')
        aug_frame = numpy.random.uniform(0,1,num_frame)
        aug_frame = numpy.where(aug_frame>0.5,1,0)
    else:
        aug_frame=numpy.zeros((num_frame,),dtype='uint8')
    # for i in range(gr_uvd.shape[0]):
    for i in numpy.random.randint(0,gr_uvd.shape[0],30):
        cur_gt_uvd = gr_uvd[i,:,:]
        cur_pred_uvd = pred_uvd[i,:,:]
        mean_pred_uvd = numpy.mean(cur_pred_uvd,axis=0)

        if aug_frame[i]:
            mean_pred_uvd+= numpy.random.normal(loc=0,scale=0.005,size=3)
            rot=numpy.random.normal(loc=0,scale=20,size=1)

            print(rot)
        else:

            rot=0

        #mean_pred_uvd is the new normcenter

        offset_uvd = cur_gt_uvd-numpy.expand_dims(mean_pred_uvd,axis=0)

        "depth translation"
        loc = numpy.where(new_r0[i,:,:,0]<1.0)
        new_r0[i,:,:,0][loc]-=mean_pred_uvd[2]
        loc = numpy.where(new_r0[i,:,:,0]>1.0)
        new_r0[i,:,:,0][loc]=1.0

        "2D translation"
        tx=-mean_pred_uvd[0]*96#cols
        ty=-mean_pred_uvd[1]*96#rows

        "scale change together with inplane rotation change"
        #size of the norm image is in proportion of the center depth value
        # pred_scale=1/uvd_hand_center[2]
        # now_scale=1/mean_pred_uvd[2]
        scale_ratio = (mean_pred_uvd[2]*300+uvd_hand_center[i,2])/uvd_hand_center[i,2]
        print(uvd_hand_center[i,2],mean_pred_uvd[2],scale_ratio)

        M = numpy.float32([[1,0,tx],[0,1,ty]])
        dst = cv2.warpAffine(new_r0[i,:,:,0],M,(96,96),borderValue=1)
        # dst1 = cv2.warpAffine(r0[i],M,(48,48),borderValue=0)

        M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],1.8*scale_ratio)
        dst= cv2.warpAffine(dst,M,(96,96),borderValue=1)


        for j in range(offset_uvd.shape[0]):
            offset_uvd[j,0:2] = (numpy.dot(M,numpy.array([offset_uvd[j,0]*96+48,offset_uvd[j,1]*96+48,1]))-48)/96

        target[i]=offset_uvd

        crop0[i,:,:,0]=dst
        crop1[i,:,:,0] = resize(crop0[i,:,:,0], (48,48), order=3,preserve_range=True)
        crop2[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)
        # target[i,0:2]=(tmp_gt_uv-tmp_pred_uv)/96.0

        fig = plt.figure()
        ax =fig.add_subplot(221)
        ax.imshow(crop0[i,:,:,0],'gray')
        ax.scatter(offset_uvd[:,0]*96+48,offset_uvd[:,1]*96+48,c='b')
        ax.scatter(48,48,c='r')
        # ax.scatter(tmp_pred_uv[0],tmp_pred_uv[1],c='r')
        #
        ax =fig.add_subplot(222)
        ax.imshow(dst,'gray')
        ax.scatter(offset_uvd[:,0]*96+48,offset_uvd[:,1]*96+48,c='b')
        ax.scatter(48,48,c='r')

        ax =fig.add_subplot(223)
        ax.imshow(crop1[i,:,:,0],'gray')
        ax.scatter(offset_uvd[:,0]*48+24,offset_uvd[:,1]*48+24,c='b')
        ax.scatter(24,24,c='r')

        ax =fig.add_subplot(224)
        ax.imshow(r0[i,:,:,0],'gray')
        plt.scatter(pred_uvd[i,:,0]*96+48,pred_uvd[i,:,1]*96+48,c='b')
        plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        plt.show()

    return crop0,crop1,target*10


def load_data(save_dir,dataset):

    f = h5py.File('%s/source/%s_crop_norm_v1.h5'%(save_dir,dataset), 'r')
    valid_idx =f['valid_idx'][...]
    train_x0 = f['img0'][...][valid_idx]
    train_y= f['uvd_norm_gt'][...][:,palm_idx,:][valid_idx]
    f.close()

    normuvd = numpy.load("%s/hier/result/%s_normuvd_palm_s0_rot_scale_ker32_lr0.000100.npy"%(save_dir,dataset))
    print(dataset, ' train_x0.shape,train_y.shape,normuvd.shape',train_x0.shape,train_y.shape,normuvd.shape)
    return numpy.expand_dims(train_x0,axis=-1), train_y,normuvd


if __name__=='__main__':
    # source_dir='/media/Data/Qi/data'
    save_dir = 'F:/HuaweiProj/data/mega'

    palm_idx =[0,1,5,9,13,17]

    dataset='test'
    f = h5py.File('%s/source/%s_crop_norm_vassi.h5'%(save_dir,dataset), 'r')

    test_x0 = f['img0'][...]
    test_y= f['uvd_norm_gt'][...][:,palm_idx,:]
    uvd_hand_center= f['uvd_hand_centre'][...]
    f.close()
    print(uvd_hand_center.shape)

    test_pred_palm = numpy.load("%s/hier/result/%s_normuvd_tmp_vass_palm_s0_rot_scale_ker32_lr0.000100.npy"%(save_dir,dataset))


    test_x0 = numpy.expand_dims(test_x0,axis=-1)
    # test_x0,test_y,test_pred_palm = load_data(save_dir,)
    get_derot_images(test_x0,test_y,test_pred_palm,uvd_hand_center,if_aug=False)
