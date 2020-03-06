__author__ = 'QiYE'


from keras.models import Model,model_from_json
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,BatchNormalization,Dense,Flatten,Dropout

def get_unet_for_regression(img_rows,img_cols,num_kern,kernel_size_1,activation,num_f1,num_f2,cnn_out_dim):
    inputs = Input((img_rows, img_cols, 1),name='input')
    conv1 = Conv2D(num_kern[0], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(inputs)
    conv1 = Conv2D(num_kern[0], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv1)#64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_kern[1], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool1)
    conv2 = Conv2D(num_kern[1], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv2)#128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(num_kern[2], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool2)
    conv3 = Conv2D(num_kern[2], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv3)#256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(num_kern[3], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool3)
    conv4 = Conv2D(num_kern[3], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv4)#512
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(num_kern[4], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool4)
    conv5 = Conv2D(num_kern[4], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv5)#1024

    up6 = concatenate([Conv2DTranspose(num_kern[5], (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(num_kern[5], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up6)#512
    conv6 = Conv2D(num_kern[5], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(num_kern[6], (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(num_kern[6], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up7)
    conv7 = Conv2D(num_kern[6], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv7)#256

    up8 = concatenate([Conv2DTranspose(num_kern[7], (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(num_kern[7], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up8)
    conv8 = Conv2D(num_kern[7], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv8)#128

    up9 = concatenate([Conv2DTranspose(num_kern[8], (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    up9=BatchNormalization()(up9)

    conv9 = Conv2D(num_kern[8], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up9)#64
    pool9 = MaxPooling2D(pool_size=(4, 4))(conv9)
    conv9 = Conv2D(num_kern[8], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool9)
    pool9 = MaxPooling2D(pool_size=(2, 2))(conv9)
    conv9 = Conv2D(num_kern[8], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool9)
    pool9 = MaxPooling2D(pool_size=(2, 2))(conv9)

    convout = Flatten()(pool9)
    fullconnet1 = Dense(num_f1,activation=activation)(convout)
    fullconnet1 = Dropout(0.3)(fullconnet1)
    fullconnet11 = Dense(num_f2,activation=activation)(fullconnet1)
    fullconnet11 = Dropout(0.3)(fullconnet11)
    reg_out = Dense(cnn_out_dim,activation=None)(fullconnet11)

    model = Model(inputs=[inputs], outputs=[reg_out])
    # print(model.summary())
    # print(model.get_config())
    return model



def get_unet_for_classification(img_rows,img_cols,num_kern,kernel_size_1,activation,num_classes):
    inputs = Input((img_rows, img_cols, 1),name='input')
    conv1 = Conv2D(num_kern[0], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(inputs)
    conv1 = Conv2D(num_kern[0], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv1)#64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_kern[1], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool1)
    conv2 = Conv2D(num_kern[1], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv2)#128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(num_kern[2], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool2)
    conv3 = Conv2D(num_kern[2], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv3)#256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(num_kern[3], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool3)
    conv4 = Conv2D(num_kern[3], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv4)#512
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(num_kern[4], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool4)
    conv5 = Conv2D(num_kern[4], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv5)#1024

    up6 = concatenate([Conv2DTranspose(num_kern[5], (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(num_kern[5], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up6)#512
    conv6 = Conv2D(num_kern[5], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(num_kern[6], (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(num_kern[6], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up7)
    conv7 = Conv2D(num_kern[6], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv7)#256


    up8 = concatenate([Conv2DTranspose(num_kern[7], (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(num_kern[7], (kernel_size_1,kernel_size_1), activation='relu', padding='same')(up8)
    conv8 = Conv2D(num_kern[7], (kernel_size_1,kernel_size_1), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(num_kern[8], (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(num_kern[8], (kernel_size_1,kernel_size_1), activation='relu', padding='same')(up9)
    conv9 = Conv2D(num_kern[8], (kernel_size_1,kernel_size_1), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation=None,name='output')(conv9)

    # out = Conv2DTranspose(num_classes, (2, 2), strides=(2, 2), padding='same')(fconv3)
    model = Model(inputs=[inputs], outputs=[conv10])
    return model





def get_unet_for_regression_heatmap(img_rows,img_cols,num_kern,kernel_size_1,activation,num_classes):
    inputs = Input((img_rows, img_cols, 1),name='input')
    conv1 = Conv2D(num_kern[0], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(inputs)
    conv1 = Conv2D(num_kern[0], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv1)#64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_kern[1], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool1)
    conv2 = Conv2D(num_kern[1], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv2)#128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(num_kern[2], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool2)
    conv3 = Conv2D(num_kern[2], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv3)#256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(num_kern[3], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool3)
    conv4 = Conv2D(num_kern[3], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv4)#512
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(num_kern[4], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool4)
    conv5 = Conv2D(num_kern[4], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv5)#1024

    up6 = concatenate([Conv2DTranspose(num_kern[5], (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(num_kern[5], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up6)#512
    conv6 = Conv2D(num_kern[5], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(num_kern[6], (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(num_kern[6], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up7)
    conv7 = Conv2D(num_kern[6], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv7)#256


    up8 = concatenate([Conv2DTranspose(num_kern[7], (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(num_kern[7], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up8)
    conv8 = Conv2D(num_kern[7], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(num_kern[8], (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(num_kern[8], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(up9)
    pool9 = MaxPooling2D(pool_size=(2, 2))(conv9)
    conv9 = Conv2D(num_kern[8], (kernel_size_1,kernel_size_1), activation=activation, padding='same')(pool9)
    pool9 = MaxPooling2D(pool_size=(2, 2))(conv9)

    conv10 = Conv2D(num_classes, (1,1),name='output')(pool9)
    # conv10 = Conv2D(num_classes, (kernel_size_1,kernel_size_1),activation='relu',name='output')(pool9)

    model = Model(inputs=[inputs], outputs=[conv10])
    print(model.summary())
    return model




