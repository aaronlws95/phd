__author__ = 'QiYE'
import matplotlib.pyplot as plt
from keras.models import Model,model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import h5py
import keras

import numpy
import os
import tensorflow as tf
from ..Model import unet
from ..preprocess import load_data

from PIL import Image
from skimage.transform import resize
from sklearn.utils import shuffle
import time

from ..utils import show_blend_img

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# source_dir = '/media/Data/Qi/data/BigHand_Challenge/Training/'
# save_dir = '/media/Data/Qi/data'

#
source_dir = 'F:/BigHand_Challenge/Training'
save_dir = 'F:/HuaweiProj/HuaWei_Seconddelivery_20180122/data/mega'

#
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.2):
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
version = 'pixel_aug_ker%d_lr%f'%(num_kern[0],lr)
batch_size=32


def cost_sigmoid(y_true, y_pred):
    sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    # sum_square = K.sum(K.reshape(K.pow((y_true-y_pred),2),[-1,num,dim]),axis=-1)
    # return K.mean(K.sum(K.sum(K.sum(sig,axis=-1),axis=-1),axis=-1))
    return K.mean(sig)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses=[]


    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        numpy.save('%s/detector_aug/loss_history_%s'%(save_dir,version),[self.losses,self.val_losses])

    # def on_batch_end(self, batch, logs={}):
    #     print(' loss ',batch)
def load_test_data():

    test_x0=numpy.empty((95744,128,160,1),dtype='float32')
    test_mask=numpy.empty((95744,128,160,1),dtype='uint8')


    f = h5py.File('%s/source/test_mask.h5'%save_dir, 'r')
    test_x0[:,4:test_x0.shape[1]-4,:,0] = f['x'][...]/2000.0
    test_mask[:,4:test_x0.shape[1]-4,:,0] = f['mask'][...]
    f.close()

    print( test_x0.shape)

    return test_x0,test_mask

def load_train_data():


    f = h5py.File('%s/source/train_mask.h5'%save_dir, 'r')
    filename = f['filename'][...]
    uvd = f['uvd'][...]
    f.close()

    print( filename.shape[0])

    return filename,uvd


def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # with tf.device('/gpu:1'):
    model = unet.get_unet_for_classification(img_rows=128,img_cols=160,
                                             num_kern=num_kern,kernel_size_1=3,
                                             activation='relu',num_classes=1)
    model.compile(optimizer=Adam(lr=lr),loss=cost_sigmoid)
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s/detector_aug/%s.json"%(save_dir,version), "w") as json_file:
        json_file.write(model_json)
    history = LossHistory()
    model_checkpoint = ModelCheckpoint('%s/detector_aug/weight_%s.h5'%(save_dir,version),monitor='val_loss',
                                       save_best_only=True,save_weights_only=True)

    uvd_jnt_gt,_,file_name=load_data.get_filenames_labels(dataset_dir=source_dir)
    num_img=len(file_name)

    idx = shuffle(numpy.arange(num_img),random_state=0)
    img_idx_train = idx[:int(num_img*0.9)]
    img_idx_test = idx[int(num_img*0.9):]
    print('total number of images', img_idx_train.shape[0],img_idx_test.shape[0],'batch',img_idx_train.shape[0]/batch_size,img_idx_test.shape[0]/batch_size)


    # 26917,2991
    model.fit_generator(
        load_data.generate_downsample_img_mask_from_file_unet_aug
        (path=source_dir,img_file_name=file_name[img_idx_train],uvd=uvd_jnt_gt[img_idx_train],batch_size=batch_size),
        steps_per_epoch=600, nb_epoch=1000,
        callbacks=[model_checkpoint,history],
        validation_data=load_data.generate_downsample_img_mask_from_file_unet_aug
        (path=source_dir,img_file_name=file_name[img_idx_test],uvd=uvd_jnt_gt[img_idx_test],batch_size=batch_size),
        validation_steps=299,max_queue_size=10)

def sigmoid(x):
    return  1.0 / (1 + numpy.exp(-x+_EPSILON))
def show_output():
    version = 'pixel_ker32_lr0.001000'
    f = h5py.File('%s/source/test_detector.h5'%save_dir, 'r')
    test_x0 = f['x'][...]
    f.close()

    f = h5py.File('%s/source/test_mask.h5'%save_dir, 'r')
    test_mask = f['mask'][...]
    f.close()
    for i in numpy.random.randint(0,10000,100):
        plt.figure()
        plt.imshow(test_x0[i,:,:,0],'gray')
        plt.figure()
        plt.imshow(test_mask[i,:,:],'gray')
        plt.show()

    mask = numpy.load("%s/detector/best/cnnout_%s.npy"%(save_dir,version))
    mask = sigmoid(mask)

    for i in numpy.random.randint(0,mask.shape[0],1000):

        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.imshow(mask[i,:,:,0],'gray')
        ax = fig.add_subplot(132)
        ax.imshow(test_mask[i,:,:],'gray')
        ax = fig.add_subplot(133)
        ax.imshow(test_x0[i,:,:,0],'gray')
        plt.show()

def predict():
    # version = 'testtest_pixel_ker32_lr0.001000'
    print(version)
    # load json and create model
    json_file = open("%s/detector_aug/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector_aug/weight_%s.h5"%(save_dir,version))
    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=cost_sigmoid)


    for i in numpy.random.randint(0,2965,1000):
        roiDepth = Image.open('F:/BigHand_Challenge/frame/images/image_D%08d.png' %(i))
        roiDepth = numpy.asarray(roiDepth, dtype='uint16')
        depth=numpy.zeros((1,128,160,1))
        depth[0,4:124,:,0] = resize(roiDepth,(120,160), order=3,preserve_range=True)/2000.0

        mask = loaded_model.predict(x=depth,batch_size=1)
        mask_up4=resize(mask[0,4:124,:,0],roiDepth.shape, order=3,preserve_range=True)
        mask_up4 = sigmoid(mask_up4)

        backimg,colors, cmap = show_blend_img.show_two_imgs(backimg=roiDepth,topimg=mask_up4,alpha=0.2)
        fig, ax = plt.subplots()
        ax.imshow(backimg,'gray')
        ax.imshow(colors,cmap=cmap)
        plt.savefig('F:/HuaweiProj/HuaWei_Seconddelivery_20180122/data/mega/detector_aug/output_img/image_D%08d.jpg' %(i))
        plt.close()

def predict_free_hand():
    # version = 'testtest_pixel_ker32_lr0.001000'
    # load json and create model
    json_file = open("%s/detector_aug/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector_aug/weight_%s.h5"%(save_dir,version))

    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=cost_sigmoid)

    f = h5py.File('%s/source/test_mask.h5'%save_dir, 'r')
    test_x0 = f['x'][...]
    f.close()
    for i in numpy.random.randint(0,test_x0.shape[0],300):

        depth=numpy.zeros((1,128,160,1))
        depth[0,4:124,:,0] = test_x0[i,:,:]

        mask = loaded_model.predict(x=depth,batch_size=1)
        print(mask.shape)
        mask = sigmoid(mask)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(mask[0,:,:,0],'gray')
        # ax.scatter(u,v)
        ax = fig.add_subplot(122)
        ax.imshow(depth[0,:,:,0],'gray')
        plt.savefig('F:/HuaweiProj/data/mega/detector/best/pixel_free_hand/image_D%08d.jpg' %(i))



def check_load_data():


    uvd_jnt_gt,_,file_name=load_data.get_filenames_labels(dataset_dir=source_dir)
    num_img=len(file_name)
    print('total number of images', num_img)
    idx = shuffle(numpy.arange(num_img),random_state=0)
    img_idx_train = idx[:int(num_img*0.9)]
    img_idx_test = idx[int(num_img*0.9):]

    load_data.generate_downsample_img_mask_from_file_unet_aug(path=source_dir,img_file_name=file_name[img_idx_test],uvd=uvd_jnt_gt[img_idx_test],batch_size=8)


def test_time():
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # with tf.device('/gpu:1'):
    model = unet.get_unet_for_classification(img_rows=128,img_cols=160,
                                             num_kern=num_kern,kernel_size_1=3,
                                             activation='relu',num_classes=1)
    model.compile(optimizer=Adam(lr=lr),loss=cost_sigmoid)

    # uvd_jnt_gt,xyz,file_name=get_filenames_labels(dataset_dir=source_dir)
    start=time.clock()
    for i in numpy.random.randint(0,1000,1000):
        depth=numpy.zeros((1,128,160,1))
        mask = model.predict(x=depth,batch_size=1)


    print(time.clock()-start)


if __name__ == '__main__':
    # test_time()
    # check_load_data()
    # train()
    predict()
    # show_output()
    # predict_free_hand()
    # predict_hand_object()
#
