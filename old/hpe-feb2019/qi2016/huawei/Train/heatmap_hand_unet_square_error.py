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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

source_dir = '/media/Data/Qi/data/BigHand_Challenge/Training/'
save_dir = '/media/Data/Qi/data'

#
# source_dir = 'F:/BigHand_Challenge/Training'
# save_dir = 'F:/HuaweiProj/data/mega'

#
# import keras.backend.tensorflow_backend as KTF
#
# def get_session(gpu_fraction=0.2):
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
version = 'heatmap_ker%d_lr%f'%(num_kern[0],lr)

def cost_sigmoid(y_true, y_pred):
    sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    # sum_square = K.sum(K.reshape(K.pow((y_true-y_pred),2),[-1,num,dim]),axis=-1)
    return K.mean(K.sum(K.sum(K.sum(sig,axis=-1),axis=-1),axis=-1))


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
def load_train_test_data():

    f = h5py.File('%s/source/test_detector.h5'%save_dir, 'r')
    train_x0 = f['x'][...]
    train_y= f['y'][...]
    f.close()

    f = h5py.File('%s/source/test_detector.h5'%save_dir, 'r')
    test_x0 = f['x'][...]
    test_y= f['y'][...]
    f.close()
    print(test_x0.shape)

    return train_x0,train_y,test_x0,test_y

def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    # with tf.device('/gpu:1'):
    model = unet.get_unet_for_regression_heatmap(img_rows=128,img_cols=160,
                                             num_kern=num_kern,kernel_size_1=3,
                                             activation='relu',num_classes=1)
    model.compile(optimizer=Adam(lr=lr),loss=cost_sigmoid)
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s/detector/%s.json"%(save_dir,version), "w") as json_file:
        json_file.write(model_json)

    history = LossHistory()
    model_checkpoint = ModelCheckpoint('%s/detector/weight_%s.h5'%(save_dir,version),monitor='val_loss',
                                       save_best_only=True,save_weights_only=True)

    train_x0,train_y,test_x0,test_y = load_train_test_data()
    # 26917,2991
    model.fit(x=train_x0, y=train_y, batch_size=128, epochs=100, verbose=1, callbacks=[model_checkpoint,history],
            validation_split=0., validation_data=(test_x0,test_y), shuffle=True)


def predict():
    version = 'heatmap_ker0_lr32.000000'
    # load json and create model
    json_file = open("%s/detector/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector/weight_%s.h5"%(save_dir,version))


    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=cost_sigmoid)

    f = h5py.File('%s/source/test_detector.h5'%save_dir, 'r')
    test_x0 = f['x'][...]
    test_y= f['y'][...]
    f.close()
    print(test_x0.shape)

    # out = loaded_model.evaluate(x=test_x0,y=test_y,batch_size=128)
    # print(out)
    mask = loaded_model.predict(x=test_x0,batch_size=128)
    print(mask.shape)

    numpy.save("%s/detector/heatmap_%s.npy"%(save_dir,version),mask)
def sigmoid(x):
    return  1.0 / (1 + numpy.exp(-x+_EPSILON))
def show_output():
    version = 'heatmap_lr0.001000'
    f = h5py.File('%s/source/test_detector.h5'%save_dir, 'r')
    test_x0 = f['x'][...]
    test_y= f['y'][...]
    f.close()
    mask = numpy.load("%s/detector/heatmap_%s.npy"%(save_dir,version))
    mask = sigmoid(mask)
    #
    # for i in numpy.random.randint(0,mask.shape[0],100):
    #     loc = numpy.argmax(mask[i,:,:,0])
    #     u = loc%40
    #     v = int(loc/40)
    #     # print(loc)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(131)
    #     ax.imshow(test_y[i,:,:,0],'gray')
    #     ax.scatter(u,v)
    #
    #     ax = fig.add_subplot(132)
    #     ax.imshow(mask[i,:,:,0],'gray')
    #
    #     ax = fig.add_subplot(133)
    #     ax.imshow(test_x0[i,:,:,0],'gray')
    #     plt.show()

    for i in numpy.random.randint(0,mask.shape[0],100):
        loc = numpy.argmax(mask[i,:,:,0])
        u = loc%40*4
        v = int(loc/40)*4

        loc = numpy.argmax(test_y[i,:,:,0])
        u_gt = loc%40*4
        v_gt = int(loc/40)*4
        # print(loc)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(test_y[i,:,:,0],'gray')
        # ax.scatter(u,v)
        ax = fig.add_subplot(122)
        ax.imshow(test_x0[i,:,:,0],'gray')
        ax.scatter(u,v)
        ax.scatter(u_gt,v_gt)
        plt.show()


if __name__ == '__main__':
    # train()
    predict()
    # show_output()
#
