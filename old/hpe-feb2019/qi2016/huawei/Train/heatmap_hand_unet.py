__author__ = 'QiYE'

from keras.models import Model,model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import h5py
import keras
from sklearn.utils import shuffle
import numpy
import os
import tensorflow as tf

from ..preprocess import load_data
from ..Model import unet

os.environ["CUDA_VISIBLE_DEVICES"]="0"

source_dir = 'F:/BigHand_Challenge/Training'
save_dir = 'F:/HuaweiProj/data/mega'

#
# base_dir = '/home/qi/data/nyu'
# save_dir = '/home/qi/Projects/Proj_TF'
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
# num_kern=[32,64,128,256,512,256,128,64,32]
num_kern=[16,32,64,128,256,128,64,32,16]

# num_kern=[4,8,16,32,64,32,16,8,4]
lr=0.001
version = 'heatmap_lr%f'%lr

def cost_sigmoid(y_true, y_pred):
    sig = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    # sum_square = K.sum(K.reshape(K.pow((y_true-y_pred),2),[-1,num,dim]),axis=-1)
    return K.mean(K.sum(K.sum(sig,axis=-1),axis=-1))


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses=[]

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        numpy.save('%s/detector/loss_history_%s'%(save_dir,version),[self.losses,self.val_losses])

    # def on_batch_end(self, batch, logs={}):
    #     print(' loss ')
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
    uvd_jnt_gt,_,file_name=load_data.get_filenames_labels(dataset_dir=source_dir)
    num_img=len(file_name)
    print('total number of images', num_img)
    idx = shuffle(numpy.arange(num_img),random_state=0)
    img_idx_train = idx[:int(num_img*0.9)]
    img_idx_test = idx[int(num_img*0.9):]


    # 26917,2991
    model.fit_generator(
        load_data.generate_arrays_from_file_unet(path=source_dir,img_file_name=file_name[img_idx_train],uvd=uvd_jnt_gt[img_idx_train],batch_size=32),
        steps_per_epoch=1000, nb_epoch=10000,
        callbacks=[model_checkpoint,history],
        validation_data=load_data.generate_arrays_from_file_unet(path=source_dir,img_file_name=file_name[img_idx_test],uvd=uvd_jnt_gt[img_idx_test],batch_size=32),
        validation_steps=2991,max_queue_size=10)

def predict():
    # load json and create model
    json_file = open("../../data/v1/unetmodel.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('../../data/v1/unet_weight_le5_v1.h5')


    loaded_model.compile(optimizer=Adam(lr=1e-5), loss="categorical_crossentropy")
    print(loaded_model.summary())
    print(loaded_model.get_config())

    x0,y = get_test_data_unet(path=base_dir,dataset='test',num_imgs=400,num_classes=2,batch_size=2)

    score = loaded_model.evaluate(x=x0,y=y,batch_size=2)
    print('score',score)

    mask = loaded_model.predict(x=x0,batch_size=2)
    print(mask.shape)


    f = h5py.File("../../data/unet_weight_le5_v1_8252.h5", 'w')
    f.create_dataset("mask", data=mask)
    f.close()


if __name__ == '__main__':
    train()
    # predict()
#
