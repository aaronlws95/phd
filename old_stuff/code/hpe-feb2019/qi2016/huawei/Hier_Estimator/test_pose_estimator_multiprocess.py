__author__ = 'QiYE'
from keras.models import model_from_json
from keras.optimizers import Adam

import numpy
from skimage.transform import resize
import matplotlib.pyplot as plt
import h5py
import cv2
import os
from PIL import Image


from multiprocessing import Pool
from ..utils import math,loss,xyz_uvd,hand_utils,get_err
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

cmap = plt.cm.rainbow
colors_map = cmap(numpy.arange(cmap.N))
rng = numpy.random.RandomState(0)
num = rng.randint(0,256,(21,))
jnt_colors = colors_map[num]
# print jnt_colors.shape
markersize = 7
linewidth=2
azim =  -177
elev = -177

hand_img_size=96
hand_size=300.0
centerU=315.944855
padWidth=100
# img_dir = '/media/Data/shanxin/megahand/'
# save_dir='/media/Data/Qi/data/hier/model'
save_dir = 'F:/HuaweiProj/data/mega'
img_dir = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/MegaEgo/'
# img_dir = 'F:/HuaweiProj/data/mega'
#
#

class HandPoseEstimator(object):

    def __init__(self,save_dir,versions):
        self.models=[]
        for version in versions:
            print('load model',version)
            # load json and create model
            json_file = open("%s/hier/model/%s.json"%(save_dir,version), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("%s/hier/model/weight_%s"%(save_dir,version))
            loaded_model.compile(optimizer=Adam(lr=1e-5), loss=loss.cost_sigmoid)
            self.models.append(loaded_model)



    def get_mask(self,depthimg,model):

        depth=numpy.zeros((1,480,640,1))
        depth[0,:,:,0] = depthimg/2000.0
        mask = model.predict(x=depth,batch_size=1)
        mask = math.sigmoid(mask[0,:,:,0])
        return mask

    def hier_estimator_detector(self,depth,setname):
        detector_model=self.models[0]

        mask = self.get_mask(depthimg=depth,model=detector_model)
        loc = numpy.where(mask>0.5)

        if  loc[0].shape[0]<30:
            print('no hand in the area or hand too small')
            return numpy.ones((hand_img_size,hand_img_size),dtype='float32'),\
                   numpy.ones((hand_img_size/2,hand_img_size/2),dtype='float32'),\
                   numpy.ones((hand_img_size/4,hand_img_size/4),dtype='float32')
        depth_value = depth[loc]
        # print('loc',loc[0].shape,numpy.max(loc[1]),numpy.min(loc[1]),numpy.max(loc[1]),numpy.min(loc[1]),',depth_value',depth_value,)
        U = numpy.mean(loc[1])
        V = numpy.mean(loc[0])
        D = numpy.mean(depth_value)
        if D<10:
            print('not valid hand area')
            return numpy.ones((hand_img_size,hand_img_size),dtype='float32'),\
                   numpy.ones((hand_img_size/2,hand_img_size/2),dtype='float32'),\
                   numpy.ones((hand_img_size/4,hand_img_size/4),dtype='float32')

        bb = numpy.array([(hand_size,hand_size,numpy.mean(depth_value))])
        bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
        margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))

        depth_w_hand_only = depth.copy()
        loc_back = numpy.where(mask<0.5)
        depth_w_hand_only[loc_back]=0
        loc_back = numpy.where(numpy.logical_and(depth_w_hand_only>D+hand_size/2,depth_w_hand_only<D-hand_size/2))
        depth_w_hand_only[loc_back]=0

        tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
        tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth_w_hand_only
        if U-margin/2+padWidth<0 or U+margin/2+padWidth>tmpDepth.shape[1]-1 or V - margin/2+padWidth <0 or V+margin/2+padWidth>tmpDepth.shape[0]-1:
            print('most hand part outside the image' )
            return numpy.ones((hand_img_size,hand_img_size),dtype='float32'),\
                   numpy.ones((hand_img_size/2,hand_img_size/2),dtype='float32'),\
                   numpy.ones((hand_img_size/4,hand_img_size/4),dtype='float32')

        crop = tmpDepth[int(V-margin/2+padWidth):int(V+margin/2+padWidth),int(U-margin/2+padWidth):int(U+margin/2+padWidth)]

        norm_hand_img=numpy.ones(crop.shape,dtype='float32')
        loc_hand=numpy.where(crop>0)
        norm_hand_img[loc_hand]=(crop[loc_hand]-D)/hand_size
        r0 = resize(norm_hand_img, (hand_img_size,hand_img_size), order=3,preserve_range=True)
        r1 = resize(norm_hand_img, (hand_img_size/2,hand_img_size/2), order=3,preserve_range=True)
        r2 = resize(norm_hand_img, (hand_img_size/4,hand_img_size/4), order=3,preserve_range=True)
        r0.shape=(1,hand_img_size,hand_img_size,1)
        r1.shape=(1,int(hand_img_size/2),int(hand_img_size/2),1)
        r2.shape=(1,int(hand_img_size/4),int(hand_img_size/4),1)
        # plt.imshow(r0,'gray')
        # plt.show()
        return r0,r1,r2,numpy.array([U,V,D]).reshape(1,1,3)

    def get_crop_for_finger_part_s0(self,r0,pred_palm_uvd,jnt_uvd_in_prev_layer,if_aug=True,scale=1.8):
        num_frame=r0.shape[0]
        new_r0=r0.copy()
        rot_angle = math.get_angle_between_two_lines(line0=(pred_palm_uvd[:,3,:]-pred_palm_uvd[:,0,:])[:,0:2])

        crop0=numpy.empty((num_frame,48,48,1),dtype='float32')
        crop1 = numpy.empty((num_frame,24,24,1),dtype='float32')


        if if_aug:
            # aug_frame=numpy.ones((num_frame,),dtype='uint8')
            aug_frame = numpy.random.uniform(0,1,num_frame)
            aug_frame = numpy.where(aug_frame>0.5,1,0)
        else:
            aug_frame=numpy.zeros((num_frame,),dtype='uint8')
        for i in range(r0.shape[0]):

            cur_pred_uvd=jnt_uvd_in_prev_layer[i]
            # print(cur_pred_uvd.shape,cur_pred_uvd.shape)

            if aug_frame[i]:
                cur_pred_uvd+= numpy.random.normal(loc=0,scale=0.05,size=3)
                rot=numpy.random.normal(loc=0,scale=15,size=1)
            else:
                rot=0
            # print(cur_pred_uvd.shape)
            "2D translation"
            tx=-cur_pred_uvd[0,0]*96#cols
            ty=-cur_pred_uvd[0,1]*96#rows

            M = numpy.float32([[1,0,tx],[0,1,ty]])
            dst = cv2.warpAffine(new_r0[i,:,:,0],M,(96,96),borderValue=1)

            M = cv2.getRotationMatrix2D((48,48),rot+rot_angle[i],scale=scale)
            dst= cv2.warpAffine(dst,M,(96,96),borderValue=1)

            crop0[i,:,:,0]=dst[24:72,24:72]
            crop1[i,:,:,0] = resize(crop0[i,:,:,0], (24,24), order=3,preserve_range=True)

        return crop0,crop1

    def get_xyz_from_normuvd(self,normuvd,uvd_hand_centre,jnt_idx,setname,bbsize):
        if setname =='icvl':
            centerU=320/2
        if setname =='nyu':
            centerU=640/2
        if setname =='msrc':
            centerU=512/2
        if setname=='mega':
            centerU=315.944855
        numImg=normuvd.shape[0]

        bbsize_array = numpy.ones((numImg,3))*bbsize
        bbsize_array[:,2]=uvd_hand_centre[:,0,2]
        bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bbsize_array)
        normUVSize = numpy.array(numpy.ceil(bbox_uvd[:,0]) - centerU,dtype='int32')
        normuvd=normuvd[:numImg].reshape(numImg,len(jnt_idx),3)
        uvd = numpy.empty_like(normuvd)
        uvd[:,:,2]=normuvd[:,:,2]*bbsize
        uvd[:,:,0:2]=normuvd[:,:,0:2]*normUVSize.reshape(numImg,1,1)
        uvd += uvd_hand_centre

        xyz_pred = xyz_uvd.uvd2xyz(setname=setname,uvd=uvd)
        return xyz_pred,uvd
    def finger_estimator(self,x):
        cur_finger=x[0]
        r0=x[1]
        palm_norm_uvd=x[2]
        scale=x[3]
        finger_norm_uvd = numpy.empty((1,3,3))
        crop0,crop1 = self.get_crop_for_finger_part_s0(r0=r0,pred_palm_uvd=palm_norm_uvd,
                                                  jnt_uvd_in_prev_layer=palm_norm_uvd[:,[cur_finger+1]],
                                                  if_aug=False,scale=scale)
        offset= self.models[cur_finger*2+2].predict(x={'input0':crop0,'input1':crop1},batch_size=1).reshape(1,1,3)
        cur_jnt_norm_uvd = get_err.get_normuvd_from_offset(offset=offset,pred_palm=palm_norm_uvd,
                                                          jnt_uvd_in_prev_layer=palm_norm_uvd[:,[cur_finger+1]],scale=scale)
        # print(cur_jnt_norm_uvd)
        cur_jnt_idx=[cur_finger*4+1+1]
        finger_norm_uvd[:,0]=cur_jnt_norm_uvd
        "make prediction for dtip on cur_finger"
        crop0,crop1 = self.get_crop_for_finger_part_s0(r0=r0,pred_palm_uvd=palm_norm_uvd,
                                                  jnt_uvd_in_prev_layer=cur_jnt_norm_uvd,
                                                  if_aug=False,scale=scale)
        cur_jnt_idx=[cur_finger*4+2+1,cur_finger*4+3+1]
        offset = self.models[cur_finger*2+3].predict(x={'input0':crop0,'input1':crop1},batch_size=1).reshape(1,2,3)
        cur_jnt_norm_uvd = get_err.get_normuvd_from_offset(offset=offset,pred_palm=palm_norm_uvd,
                                                          jnt_uvd_in_prev_layer=cur_jnt_norm_uvd,scale=scale)
        finger_norm_uvd[:,1:]=cur_jnt_norm_uvd
        return finger_norm_uvd

    def pose_estimator_pool(self,depth):
        p = Pool(8)
        scale=1.8
        setname='mega'
        bbsize=300
        palm_idx=[0,1,5,9,13,17]
        "load models"
        pose_norm_uvd = numpy.empty((1,21,3))
        "prediction for palm_stage_0"

        r0,r1,r2,meanUVD = self.hier_estimator_detector(depth,setname=setname)
        palm_norm_uvd = self.models[1].predict(x={'input0':r0,'input1':r1,'input2':r2},batch_size=1).reshape(1,6,3)
        pose_norm_uvd[:,palm_idx,:]=palm_norm_uvd


        data=[(0,r0,palm_norm_uvd,scale),(1,r0,palm_norm_uvd,scale),(2,r0,palm_norm_uvd,scale),
               (3,r0,palm_norm_uvd,scale),(4,r0,palm_norm_uvd,scale)]
        start=time.clock()
        tmp = p.map(self.finger_estimator, data)
        print('pool time',1/(time.clock()-start))
        for cur_finger,jnt_uvd in enumerate(tmp):
            pose_norm_uvd[:,cur_finger*4+1+1:cur_finger*4+4+1]=jnt_uvd
        xyz_pred ,uvd_pred= self.get_xyz_from_normuvd(normuvd=pose_norm_uvd,uvd_hand_centre=meanUVD,jnt_idx=range(21),setname=setname,bbsize=bbsize)
        # print(pose_norm_uvd)
        # print(xyz_pred)
        return xyz_pred,uvd_pred



def main():
    versions=['pixel_fullimg_ker32_lr0.001000','vass_palm_s0_rot_scale_ker32_lr0.000100']
    for i in range(5):
        versions.append('pip_s0_finger%d_smalljiter_ker48_lr0.000100'%i)
        versions.append('dtip_s0_finger%d_smalljiter_ker48_lr0.000100'%i)
    estimator = HandPoseEstimator(save_dir,versions)


    dataset='test'
    # f = h5py.File('F:/HuaweiProj/data/mega/source/%s_crop_norm_v1.h5'%(dataset), 'r')
    f = h5py.File('F:/HuaweiProj/data/mega/source/%s_crop_norm_v1.h5'%(dataset), 'r')
    new_file_names = f['new_file_names'][...]
    gt_xyz= f['xyz_gt'][...]
    f.close()
    # for i in range(new_file_names.shape[0]):
    # err=[]
    # start=time.clock()
    for i in range(0,new_file_names.shape[0],1):
    # for i in numpy.random.randint(0,new_file_names.shape[0],100):
        cur_frame=new_file_names[i]
        depth = Image.open("%s%s.png"%(img_dir,cur_frame))
        depth = numpy.asarray(depth, dtype='uint16')
        # start=time.clock()
        xyz_pred,uvd_pred = estimator.pose_estimator_pool(depth)

        tmp_err=numpy.mean(numpy.sqrt(numpy.sum((xyz_pred[0]-gt_xyz[i])**2,axis=-1)))
        # print(tmp_err)
        # hand_utils.show_two_hand_skeleton(xyz_pred[0],gt_xyz[i])
        # print(tmp_err)
        # err.append(tmp_err)
        imgcopy=depth.copy()
        # msk = numpy.logical_and(2000 > imgcopy, imgcopy > 0)
        # msk2 = numpy.logical_or(imgcopy == 0, imgcopy == 1000)
        min = imgcopy.min()
        max = imgcopy.max()
        imgcopy = (imgcopy - min) / (max - min) * 255.
        imgcopy = imgcopy.astype('uint8')
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)
        # poseimg=depth.copy()
        # poseimg = numpy.zeros_like(depth)
        for j in range(uvd_pred.shape[1]):
            cv2.circle(imgcopy,(int(uvd_pred[0,j,0]),int(uvd_pred[0,j,1])), int(3000.0/numpy.mean(uvd_pred[0,j,2])), (0, 255, 0), -1)
            # cv2.circle(imgcopy,(uvd_pred[i,j,0],uvd_pred[i,j,1]),10, (255, 0, 0), -1)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', imgcopy)
        cv2.waitKey(1)




if __name__=='__main__':

    main()


