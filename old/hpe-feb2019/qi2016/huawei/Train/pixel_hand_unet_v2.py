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


os.environ["CUDA_VISIBLE_DEVICES"]="3"

source_dir = '/media/Data/Qi/data/BigHand_Challenge/Training/'
save_dir = '/media/Data/Qi/data'

#
# source_dir = 'F:/BigHand_Challenge/Training'
# save_dir = 'F:/HuaweiProj/data/mega'

#
# import keras.backend.tensorflow_backend as KTF
#
# def get_session(gpu_fraction=0.48):
#     '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
#
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# KTF.set_session(get_session())


_EPSILON = 10e-8

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
num_kern=[32,64,128,256,512,256,128,64,32]
# num_kern=[16,32,64,128,256,128,64,32,16]

# num_kern=[4,8,16,32,64,32,16,8,4]
lr=0.001
version = 'pixel_v2_ker%d_lr%f'%(num_kern[0],lr)

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
        numpy.save('%s/detector/loss_history_%s'%(save_dir,version),[self.losses,self.val_losses])

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
    with open("%s/detector/%s.json"%(save_dir,version), "w") as json_file:
        json_file.write(model_json)
    model.load_weights("%s/detector/weight_%s.h5"%(save_dir,version))
    history = LossHistory()
    model_checkpoint = ModelCheckpoint('%s/detector/weight_%s.h5'%(save_dir,version),monitor='val_loss',
                                       save_best_only=True,save_weights_only=True)

    test_x0,test_y = load_test_data()
    train_img_file_name,train_uvd = load_train_data()
    # 26917,2991
    model.fit_generator(
        load_data.generate_train(path=source_dir,img_file_name=train_img_file_name,uvd=train_uvd,batch_size=64),
        steps_per_epoch=2000, nb_epoch=10000,
        callbacks=[model_checkpoint,history], validation_data=(test_x0,test_y))


def predict():
    version = 'testtest_pixel_ker32_lr0.001000'
    # load json and create model
    json_file = open("%s/detector/best/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector/best/weight_%s.h5"%(save_dir,version))


    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=cost_sigmoid)

    f = h5py.File('%s/source/test_detector.h5'%save_dir, 'r')
    test_x0 = f['x'][...]
    f.close()
    # print(test_x0.shape)
    # test_mask=numpy.empty(test_x0.shape,dtype='uint8')
    # f = h5py.File('%s/source/test_mask.h5'%save_dir, 'r')
    # test_mask[:,4:test_x0.shape[1]-4,:,0] = f['mask'][...]
    # f.close()


    # out = loaded_model.evaluate(x=test_x0,y=test_y,batch_size=128)
    # print(out)
    mask = loaded_model.predict(x=test_x0,batch_size=128)
    print(mask.shape)

    numpy.save("%s/detector/cnnout_%s.npy"%(save_dir,version),mask)
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

def predict_hand_object():
    version = 'testtest_pixel_ker32_lr0.001000'
    print(version)
    # load json and create model
    json_file = open("%s/detector/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector/weight_%s.h5"%(save_dir,version))


    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=cost_sigmoid)


    for i in numpy.random.randint(0,2965,1000):

        roiDepth = Image.open('F:/BigHand_Challenge/hand_object/images/image_D%08d.png' %(i))
        roiDepth = numpy.asarray(roiDepth, dtype='uint16')
        depth=numpy.zeros((1,128,160,1))
        depth[0,4:124,:,0] = resize(roiDepth,(120,160), order=3,preserve_range=True)

        mask = loaded_model.predict(x=depth,batch_size=1)
        print(mask.shape)
        mask = sigmoid(mask)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(mask[0,:,:,0],'gray')
        # ax.scatter(u,v)
        ax = fig.add_subplot(122)
        ax.imshow(depth[0,:,:,0],'gray')
        plt.savefig('F:/HuaweiProj/data/mega/detector/best/pixel_hand_object/image_D%08d.jpg' %(i))

def predict_free_hand():
    version = 'testtest_pixel_ker32_lr0.001000'
    # load json and create model
    json_file = open("%s/detector/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector/weight_%s.h5"%(save_dir,version))


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



if __name__ == '__main__':
    train()
    # predict()
    # show_output()
    predict_free_hand()
    # predict_hand_object()
#
