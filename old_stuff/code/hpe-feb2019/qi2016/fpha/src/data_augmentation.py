import numpy as np
from PIL import Image
import cv2
from math import pi
from skimage.transform import resize

def augment_data_3d_rot_scale(r0,r1,r2,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_r1=r1[:,:,:,0].copy()
    new_r2=r2[:,:,:,0].copy()

    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    rot = np.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor = np.random.normal(loc=1,scale=0.05,size=num_frame)

    # for i in range(0,gr_uvd.shape[0],1):
    for i in np.random.randint(0,num_frame,int(num_frame*0.5)):
        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((48,48),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=np.dot(M,np.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96
        M = cv2.getRotationMatrix2D((24,24),rot[i],scale_factor[i])
        new_r1[i] = cv2.warpAffine(new_r1[i],M,(48,48),borderValue=1)
        M = cv2.getRotationMatrix2D((12,12),rot[i],scale_factor[i])
        new_r2[i] = cv2.warpAffine(new_r2[i],M,(24,24),borderValue=1)

    return np.expand_dims(new_r0,axis=-1), np.expand_dims(new_r1,axis=-1), np.expand_dims(new_r2,axis=-1), new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])

def get_angle_between_two_lines(line0,line1=(0,-1)):
    rot =np.arccos(np.dot(line0,line1)/np.linalg.norm(line0,axis=1))
    loc_neg = np.where(line0[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = np.cast['float32'](rot/pi*180)
    # print np.where(rot==180)[0].shape[0]
    # rot[np.where(rot==180)] =179
    return rot

def get_crop_for_finger_part_s0(r0,gr_uvd,pred_uvd,jnt_uvd_in_prev_layer,if_aug=True,aug_trans=0.05, aug_rot=15,scale=1.8):
    new_r0=r0.copy()
    num_frame=gr_uvd.shape[0]
    rot_angle = get_angle_between_two_lines(line0=(pred_uvd[:,3,:]-pred_uvd[:,0,:])[:,0:2])

    crop0=np.empty((num_frame,48,48,1),dtype='float32')
    crop1 = np.empty((num_frame,24,24,1),dtype='float32')
    target = np.empty_like(gr_uvd)

    if if_aug:
        # aug_frame=np.ones((num_frame,),dtype='uint8')
        aug_frame = np.random.uniform(0,1,num_frame)
        aug_frame = np.where(aug_frame>0.5,1,0)
    else:
        aug_frame=np.zeros((num_frame,),dtype='uint8')
    for i in range(gr_uvd.shape[0]):
        cur_gt_uvd = gr_uvd[i]
        cur_pred_uvd=jnt_uvd_in_prev_layer[i]
        # print(cur_pred_uvd.shape,cur_pred_uvd.shape)

        if aug_frame[i]:
            cur_pred_uvd+= np.random.normal(loc=0,scale=aug_trans,size=3)
            rot=np.random.normal(loc=0,scale=aug_rot,size=1)
        else:
            rot=0
        #mean_pred_uvd is the new normcenter
        offset_uvd = cur_gt_uvd-cur_pred_uvd
        # print(offset_uvd.shape)

        "2D translation"
        tx=-cur_pred_uvd[0,0]*96#cols
        ty=-cur_pred_uvd[0,1]*96#rows

        M = np.float32([[1,0,tx],[0,1,ty]])
        dst = cv2.warpAffine(new_r0[i,:,:,0],M,(96,96),borderValue=1)

        M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],scale=scale)
        dst= cv2.warpAffine(dst,M,(96,96),borderValue=1)
        for j in range(offset_uvd.shape[0]):
            offset_uvd[j,0:2] = (np.dot(M,np.array([offset_uvd[j,0]*96+48,offset_uvd[j,1]*96+48,1]))-48)/96

        target[i]=offset_uvd

        crop0[i,:,:,0]=dst[24:72,24:72]
        crop1[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)

    return crop0,crop1,target.reshape(target.shape[0],-1)
