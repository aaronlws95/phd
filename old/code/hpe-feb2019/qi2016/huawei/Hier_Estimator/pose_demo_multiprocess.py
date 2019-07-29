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

from multiprocessing import Process, Manager, Value
from ..utils import math,loss,xyz_uvd,hand_utils,get_err
import time

import copy
import pyrealsense2 as rs


os.environ["CUDA_VISIBLE_DEVICES"]="0"
colors = numpy.array([[0.,0,0],
              [1.0,.0,0],
              [0.8,.0,0],
              [0.6,0,0],
              [0.4,0,0],

              [0,1,0],
              [0,0.8,0],
              [0,0.6,0],
              [0,0.4,0],

              [0,0,1],
              [0,0,0.8],
              [0,0,0.6],
              [0,0,0.4],

              [1,1,0],
              [1,0.8,0],
              [1.,0.6,0],
              [1,0.4,0],

              [1,0,1],
              [1.,0,0.8],
              [1,0,0.6],
              [1,0,0.4],
              ]).reshape(21,3)*255
setname='mega'
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
save_dir = 'F:/HuaweiProj/data/mega/'
img_dir = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/MegaEgo/'
# img_dir = 'F:/HuaweiProj/data/mega'
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
def get_session(gpu_fraction=0.4):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



class HandPoseEstimator(object):
    """
    Realtime pipeline for handpose estimation
    """

    # states of pipeline
    STATE_IDLE = 0
    STATE_INIT = 1
    STATE_RUN = 2

    # different hands
    HAND_LEFT = 0
    HAND_RIGHT = 1

    # different detectors
    DETECTOR_COM = 0

    def __init__(self, save_dir,versions):
        """
        Initialize data
        :param poseNet: network for pose estimation
        :param config: configuration
        :param di: depth importer
        :param verbose: print additional info
        :param comrefNet: refinement network from center of mass detection
        :return: None
        """
        self.versions=versions
        self.save_dir=save_dir
        self.sync = Manager().dict(has_hand=False,
                                   r0=numpy.ones((96, 96), dtype='float32'),
                                   r1=numpy.ones((96, 96), dtype='float32'),
                                   r2=numpy.ones((96, 96), dtype='float32'),
                                   meanUVD=numpy.asarray([0, 0, 300]),
                                   mask=numpy.ones((480, 640), dtype='float32'),
                                   depth=numpy.ones((480, 640), dtype='float32'))

    def read_and_detect(self):
        KTF.set_session(get_session())
        print("acitvate sensor" )
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        version=self.versions[0]
        print('load model',version)
        # load json and create model
        json_file = open("%s/hier/model/%s.json"%(self.save_dir,version), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("%s/hier/model/weight_%s"%(self.save_dir,version))
        model.compile(optimizer=Adam(lr=1e-5), loss=loss.cost_sigmoid)

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            # depth_image = numpy.asanyarray(depth_frame.get_data())
            depth = numpy.asarray(depth_frame.get_data(), dtype='uint16')*depth_scale*1000
            # print(depth.shape,depth.max(),depth.min(),depth.dtype)

            startd=time.clock()
            has_hand,r0,r1,r2,meanUVD,mask = self.hier_estimator_detector(depth,model,setname=setname)

            print("hand detection %d fps "%(int(1/(time.clock() - startd))))

            self.sync.update(has_hand=has_hand, r0=r0,r1=r1,r2=r2,meanUVD=meanUVD, depth=depth)

        # we are done
        print("Exiting producer...")
        pipeline.stop()
        return True

    def estimate_and_show(self):
        KTF.set_session(get_session())
        models=[]
        for version in self.versions[1:]:
            print('load model',version)
            # load json and create model
            json_file = open("%s/hier/model/%s.json"%(self.save_dir,version), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("%s/hier/model/weight_%s"%(self.save_dir,version))
            loaded_model.compile(optimizer=Adam(lr=1e-5), loss=loss.cost_sigmoid)
            models.append(loaded_model)


        while True:

            frm = copy.deepcopy(self.sync)
            startd=time.clock()
            has_hand,xyz_pred,uvd_pred = self.pose_estimator(models,has_hand=frm['has_hand'],
                                                                  r0=frm['r0'],r1=frm['r1'],r2=frm['r2'],meanUVD=frm['meanUVD'])
            print("pose estimation %d fps "%int(1/(time.clock() - startd)))
            imgcopy=frm['depth'].copy()

            min = imgcopy.min()
            max = imgcopy.max()
            imgcopy = (imgcopy - min) / (max - min) * 255.
            imgcopy = imgcopy.astype('uint8')
            imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)
            if has_hand:
                imgcopy=show_2D_hand_skeleton(imgcopy=imgcopy,uvd_pred=numpy.array(uvd_pred,dtype='int32'))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', imgcopy)
            cv2.waitKey(1)

        cv2.destroyAllWindows()
        # we are done
        print("Exiting pose estimator...")
        return True


    def get_mask(self,depthimg,model):
        # model=self.models[0]
        depth=numpy.zeros((1,480,640,1))
        depth[0,:,:,0] = depthimg/2000.0
        mask = model.predict(x=depth,batch_size=1)
        mask = math.sigmoid(mask[0,:,:,0])
        return mask

    def hier_estimator_detector(self,depth,model,setname):

        mask = self.get_mask(depthimg=depth,model=model)
        loc = numpy.where(mask>0.5)

        if  loc[0].shape[0]<30:
            print('no hand in the area or hand too small')
            return False,numpy.ones((1,hand_img_size,hand_img_size,1),dtype='float32'),\
                   numpy.ones((1,int(hand_img_size/2),int(hand_img_size/2),1),dtype='float32'),\
                   numpy.ones((1,int(hand_img_size/4),int(hand_img_size/4),1),dtype='float32'),\
                   numpy.array([depth.shape[0]/2,depth.shape[1]/2,numpy.mean(depth)]).reshape(1,1,3),mask
        depth_value = depth[loc]
        # print('loc',loc[0].shape,numpy.max(loc[1]),numpy.min(loc[1]),numpy.max(loc[1]),numpy.min(loc[1]),',depth_value',depth_value,)
        U = numpy.mean(loc[1])
        V = numpy.mean(loc[0])
        D = numpy.mean(depth_value)
        if D<10:
            print('not valid hand area')
            return False,numpy.ones((1,hand_img_size,hand_img_size,1),dtype='float32'),\
                   numpy.ones((1,int(hand_img_size/2),int(hand_img_size/2),1),dtype='float32'),\
                   numpy.ones((1,int(hand_img_size/4),int(hand_img_size/4),1),dtype='float32'),\
                   numpy.array([depth.shape[0]/2,depth.shape[1]/2,numpy.mean(depth)]).reshape(1,1,3),mask

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
            return False,numpy.ones((1,hand_img_size,hand_img_size,1),dtype='float32'),\
                   numpy.ones((1,int(hand_img_size/2),int(hand_img_size/2),1),dtype='float32'),\
                   numpy.ones((1,int(hand_img_size/4),int(hand_img_size/4),1),dtype='float32'),\
                   numpy.array([depth.shape[0]/2,depth.shape[1]/2,numpy.mean(depth)]).reshape(1,1,3),mask

        crop = tmpDepth[int(V-margin/2+padWidth):int(V+margin/2+padWidth),int(U-margin/2+padWidth):int(U+margin/2+padWidth)]

        norm_hand_img=numpy.ones(crop.shape,dtype='float32')
        loc_hand=numpy.where(crop>0)
        norm_hand_img[loc_hand]=(crop[loc_hand]-D)/hand_size
        r0 = resize(norm_hand_img, (hand_img_size,hand_img_size), order=3,preserve_range=True)
        r1 = resize(norm_hand_img, (int(hand_img_size/2),int(hand_img_size/2)), order=3,preserve_range=True)
        r2 = resize(norm_hand_img, (int(hand_img_size/4),int(hand_img_size/4)), order=3,preserve_range=True)
        r0.shape=(1,hand_img_size,hand_img_size,1)
        r1.shape=(1,int(hand_img_size/2),int(hand_img_size/2),1)
        r2.shape=(1,int(hand_img_size/4),int(hand_img_size/4),1)
        # plt.imshow(r0,'gray')
        # plt.show()
        return True,r0,r1,r2,numpy.array([U,V,D]).reshape(1,1,3),mask
    def pose_estimator(self,models,has_hand,r0,r1,r2,meanUVD):
        scale=1.8
        setname='mega'
        bbsize=300
        palm_idx=[0,1,5,9,13,17]
        "load models"
        pose_norm_uvd = numpy.empty((1,21,3))
        "prediction for palm_stage_0"

        if has_hand ==False:
            return False,-1,-1

        palm_norm_uvd = models[0].predict(x={'input0':r0,'input1':r1,'input2':r2},batch_size=1).reshape(1,6,3)
        pose_norm_uvd[:,palm_idx,:]=palm_norm_uvd
        for cur_finger in range(5):
            "make prediction for pip on cur_finger"
            crop0,crop1 = get_crop_for_finger_part_s0(r0=r0,pred_palm_uvd=palm_norm_uvd,
                                                      jnt_uvd_in_prev_layer=palm_norm_uvd[:,[cur_finger+1]],
                                                      if_aug=False,scale=scale)
            offset= models[cur_finger*2+1].predict(x={'input0':crop0,'input1':crop1},batch_size=1).reshape(1,1,3)
            cur_jnt_norm_uvd = get_err.get_normuvd_from_offset(offset=offset,pred_palm=palm_norm_uvd,
                                                              jnt_uvd_in_prev_layer=palm_norm_uvd[:,[cur_finger+1]],scale=scale)
            # print(cur_jnt_norm_uvd)
            cur_jnt_idx=[cur_finger*4+1+1]
            pose_norm_uvd[:,cur_jnt_idx]=cur_jnt_norm_uvd
            "make prediction for dtip on cur_finger"
            crop0,crop1 = get_crop_for_finger_part_s0(r0=r0,pred_palm_uvd=palm_norm_uvd,
                                                      jnt_uvd_in_prev_layer=cur_jnt_norm_uvd,
                                                      if_aug=False,scale=scale)
            cur_jnt_idx=[cur_finger*4+2+1,cur_finger*4+3+1]
            offset = models[cur_finger*2+2].predict(x={'input0':crop0,'input1':crop1},batch_size=1).reshape(1,2,3)
            cur_jnt_norm_uvd = get_err.get_normuvd_from_offset(offset=offset,pred_palm=palm_norm_uvd,
                                                              jnt_uvd_in_prev_layer=cur_jnt_norm_uvd,scale=scale)
            pose_norm_uvd[:,cur_jnt_idx]=cur_jnt_norm_uvd

        xyz_pred ,uvd_pred= get_xyz_from_normuvd(normuvd=pose_norm_uvd,uvd_hand_centre=meanUVD,jnt_idx=range(21),setname=setname,bbsize=bbsize)

        return True,xyz_pred,uvd_pred

    def processVideoThreaded(self):
        """
        Use video as input
        :param device: device id
        :return: None
        """

        print("Create image capture and hand detection process...")
        p = Process(target=self.read_and_detect, args=[])
        p.daemon = True
        print("Create hand poes estimation and show process...")
        c = Process(target=self.estimate_and_show, args=[])
        c.daemon = True
        p.start()
        c.start()

        c.join()
        p.join()
def show_2D_hand_skeleton(imgcopy,uvd_pred):

    ratio_size=int(1500.0/numpy.mean(uvd_pred[0,:,2]))
    for k in [1,5,9,13,17]:

        cv2.line(imgcopy,tuple(uvd_pred[0,0,0:2]), tuple(uvd_pred[0,k,0:2]), tuple(colors[k]), ratio_size)
        cv2.line(imgcopy,tuple(uvd_pred[0,k,0:2]), tuple(uvd_pred[0,k+1,0:2]), tuple(colors[k+1]), ratio_size)
        cv2.line(imgcopy,tuple(uvd_pred[0,k+1,0:2]), tuple(uvd_pred[0,k+2,0:2]), tuple(colors[k+2]), ratio_size)
        cv2.line(imgcopy,tuple(uvd_pred[0,k+2,0:2]), tuple(uvd_pred[0,k+3,0:2]), tuple(colors[k+3]), ratio_size)
    ratio_size=int(3000.0/numpy.mean(uvd_pred[0,:,2]))
    for j in range(uvd_pred.shape[1]):
        cv2.circle(imgcopy,(int(uvd_pred[0,j,0]),int(uvd_pred[0,j,1])), ratio_size, tuple(colors[j]), -1)
    return imgcopy


def load_model(save_dir,version):

    print('load model',version)
    # load json and create model
    json_file = open("%s/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/weight_%s"%(save_dir,version))
    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=loss.cost_sigmoid)
    return loaded_model
def get_crop_for_finger_part_s0(r0,pred_palm_uvd,jnt_uvd_in_prev_layer,if_aug=True,scale=1.8):
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

def get_xyz_from_normuvd(normuvd,uvd_hand_centre,jnt_idx,setname,bbsize):
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




def main():
    versions=['pixel_fullimg_ker32_lr0.001000','vass_palm_s0_rot_scale_ker32_lr0.000100']
    for i in range(5):
        versions.append('pip_s0_finger%d_smalljiter_ker48_lr0.000100'%i)
        versions.append('dtip_s0_finger%d_smalljiter_ker48_lr0.000100'%i)
    instance = HandPoseEstimator(save_dir,versions)
    instance.processVideoThreaded()

if __name__=='__main__':

    main()


