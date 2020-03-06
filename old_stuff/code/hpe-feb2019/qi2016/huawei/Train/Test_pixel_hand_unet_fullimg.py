__author__ = 'QiYE'
import time
from keras.models import Model,model_from_json
from keras.optimizers import Adam

from keras import backend as K
import numpy
import os
import tensorflow as tf
from ..utils import xyz_uvd,show_blend_img
from ..preprocess import load_data



import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
os.environ["CUDA_VISIBLE_DEVICES"]="0"

source_dir = 'F:/BigHand_Challenge/Training'
save_dir = 'F:/HuaweiProj/HuaWei_Seconddelivery_20180122/data/mega'

#
# base_dir = '/home/qi/data/nyu'
# save_dir = '/home/qi/Projects/Proj_TF'
#
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.03):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())


_EPSILON = 10e-8

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
num_kern=[32,64,128,256,512,256,128,64,32]
# num_kern=[16,32,64,128,256,128,64,32,16]

# num_kern=[4,8,16,32,64,32,16,8,4]
lr=0.001
version = 'pixel_fullimg_ker%d_lr%f'%(num_kern[0],lr)

def cost_sigmoid(y_true, y_pred):
    sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    # sum_square = K.sum(K.reshape(K.pow((y_true-y_pred),2),[-1,num,dim]),axis=-1)
    return K.mean(sig)

def get_filenames_labels(dataset_dir):

    xyz_jnt_gt=[]
    file_name = []
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
    with open('%s/Annotation.txt'%(dataset_dir), mode='r',encoding='utf-8',newline='') as f:
        for line in f:
            part = line.split('\t')
            file_name.append(part[0])
            xyz_jnt_gt.append(part[1:64])
    f.close()
    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float64')

    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt =xyz_uvd.xyz2uvd(xyz=xyz_jnt_gt,setname='mega')

    return uvd_jnt_gt,xyz_jnt_gt,numpy.array(file_name)

def sigmoid(x):
    return  1.0 / (1 + numpy.exp(-x+_EPSILON))
def predict_hand_object():
    version = 'pixel_fullimg_ker32_lr0.001000'
    print(version)
    # load json and create model
    json_file = open("%s/detector/best/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector/best/weight_%s.h5"%(save_dir,version))

    # uvd_jnt_gt,_,file_name=load_data.get_filenames_labels(dataset_dir=source_dir)
    # num_img=len(file_name)
    # print('total number of images', num_img)
    # img_idx_test = numpy.arange(num_img)


    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=cost_sigmoid)

    # uvd_jnt_gt,xyz,file_name=get_filenames_labels(dataset_dir=source_dir)
    for i in numpy.random.randint(0,2965,100):

        roiDepth = Image.open('F:/BigHand_Challenge/hand_object/images/image_D%08d.png' %(i))
        depth=numpy.zeros((1,480,640,1))
        depth[0,:,:,0] = numpy.asarray(roiDepth, dtype='uint16')/2000.0

        mask = loaded_model.predict(x=depth,batch_size=1)
        print(mask.shape)
        mask = sigmoid(mask[0,:,:,0])
        backimg,colors, cmap = show_blend_img.show_two_imgs(backimg=depth[0,:,:,0],topimg=mask,alpha=0.2)
        fig, ax = plt.subplots()
        ax.imshow(backimg,'gray')
        ax.imshow(colors,cmap=cmap)
        # plt.show()

        plt.savefig('F:/HuaweiProj/data/mega/detector/best/pixel_hand_object_fullimg/image_D%08d.jpg' %(i))

def predict_free_hand():
    version = 'pixel_fullimg_ker32_lr0.001000'
    print(version)
    # load json and create model
    json_file = open("%s/detector/best/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector/best/weight_%s.h5"%(save_dir,version))

    # uvd_jnt_gt,_,file_name=load_data.get_filenames_labels(dataset_dir=source_dir)
    # num_img=len(file_name)
    # print('total number of images', num_img)
    # img_idx_test = numpy.arange(num_img)


    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=cost_sigmoid)

    # uvd_jnt_gt,xyz,file_name=get_filenames_labels(dataset_dir=source_dir)
    for i in numpy.random.randint(0,296500,100):

        roiDepth = Image.open('F:/BigHand_Challenge/frame/images/image_D%08d.png' %(i))
        depth=numpy.zeros((1,480,640,1))
        depth[0,:,:,0] = numpy.asarray(roiDepth, dtype='uint16')/2000.0

        mask = loaded_model.predict(x=depth,batch_size=1)
        print(mask.shape)
        mask = sigmoid(mask[0,:,:,0])
        backimg,colors, cmap = show_blend_img.show_two_imgs(backimg=depth[0,:,:,0],topimg=mask,alpha=0.2)
        fig, ax = plt.subplots()
        ax.imshow(backimg,'gray')
        ax.imshow(colors,cmap=cmap)
        # plt.show()
        plt.savefig('F:/HuaweiProj/data/mega/detector/best/pixel_free_hand_fullimg/image_D%08d.jpg' %(i))


def test_time():
    version = 'pixel_fullimg_ker32_lr0.001000'
    print(version)
    # load json and create model
    json_file = open("%s/hier/model/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)


    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=cost_sigmoid)

    # uvd_jnt_gt,xyz,file_name=get_filenames_labels(dataset_dir=source_dir)
    start=time.clock()
    for i in numpy.random.randint(0,1000,1000):
        depth=numpy.zeros((1,480,640,1))
        mask = loaded_model.predict(x=depth,batch_size=1)

    print(time.clock()-start)

if __name__ == '__main__':
    # train()
    # predict_free_hand()
    test_time()
    # predict_hand_object()
#
