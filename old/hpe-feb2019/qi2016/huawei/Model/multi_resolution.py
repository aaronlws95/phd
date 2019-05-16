__author__ = 'QiYE'


from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D,BatchNormalization,Dense,Flatten,Dropout

def get_multi_reso_for_palm_stage_0(img_rows,img_cols,num_kern,num_f1,num_f2,cnn_out_dim):
    padding='valid'
    activation='relu'
    inputs_r0 = Input((img_rows, img_cols, 1),name='input0')
    conv1 = Conv2D(num_kern[0], (5,5), activation=activation, padding=padding)(inputs_r0)
    pool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

    conv2 = Conv2D(num_kern[1], (3,3), activation=activation, padding=padding)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(num_kern[2], (3,3), activation=activation, padding=padding)(pool2)
    conv_r0 = MaxPooling2D(pool_size=(2, 2))(conv3)

    inputs_r1 = Input((img_rows/2, img_cols/2, 1),name='input1')
    conv1 = Conv2D(num_kern[0], (5,5), activation=activation, padding=padding)(inputs_r1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_kern[1], (3,3), activation=activation, padding=padding)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(num_kern[2], (3,3), activation=activation, padding=padding)(pool2)
    conv_r1 = MaxPooling2D(pool_size=(2, 2))(conv3)

    inputs_r2 = Input((img_rows/4, img_cols/4, 1),name='input2')
    conv1 = Conv2D(num_kern[0], (5,5), activation=activation, padding=padding)(inputs_r2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_kern[1], (5,5), activation=activation, padding=padding)(pool1)
    pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)

    conv3 = Conv2D(num_kern[2], (3,3), activation=activation, padding=padding)(pool2)
    conv_r2 = MaxPooling2D(pool_size=(1, 1))(conv3)

    batch_norm = concatenate([conv_r0,conv_r1,conv_r2], axis=3)
    batch_norm=BatchNormalization()(batch_norm)

    convout = Flatten()(batch_norm)


    fullconnet1 = Dense(num_f1,activation=activation)(convout)
    fullconnet1 = Dropout(0.3)(fullconnet1)
    fullconnet11 = Dense(num_f2,activation=activation)(fullconnet1)
    fullconnet11 = Dropout(0.3)(fullconnet11)
    reg_out = Dense(cnn_out_dim,activation=None)(fullconnet11)
    model = Model(inputs=[inputs_r0,inputs_r1,inputs_r2], outputs=[reg_out])
    print(model.summary())
    return model


def get_multi_reso_for_finger_old(img_rows,img_cols,num_kern,num_f1,num_f2,cnn_out_dim):
    padding='valid'
    activation='relu'
    inputs_r0 = Input((img_rows, img_cols, 1),name='input0')
    conv1 = Conv2D(num_kern[0], (5,5), activation=activation, padding=padding)(inputs_r0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_kern[1], (3,3), activation=activation, padding=padding)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(num_kern[2], (3,3), activation=activation, padding=padding)(pool2)
    conv_r0 = MaxPooling2D(pool_size=(2, 2))(conv3)

    inputs_r1 = Input((img_rows/2, img_cols/2, 1),name='input1')
    conv1 = Conv2D(num_kern[0], (5,5), activation=activation, padding=padding)(inputs_r1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_kern[1], (5,5), activation=activation, padding=padding)(pool1)
    pool2 = MaxPooling2D(pool_size=(1, 1))(conv2)

    conv3 = Conv2D(num_kern[2], (3,3), activation=activation, padding=padding)(pool2)
    conv_r1 = MaxPooling2D(pool_size=(1, 1))(conv3)


    batch_norm = concatenate([conv_r0,conv_r1], axis=3)
    batch_norm=BatchNormalization()(batch_norm)

    convout = Flatten()(batch_norm)

    fullconnet1 = Dense(num_f1,activation=activation)(convout)
    fullconnet1 = Dropout(0.3)(fullconnet1)
    fullconnet11 = Dense(num_f2,activation=activation)(fullconnet1)
    fullconnet11 = Dropout(0.3)(fullconnet11)
    reg_out = Dense(cnn_out_dim,activation=None)(fullconnet11)
    model = Model(inputs=[inputs_r0,inputs_r1], outputs=[reg_out])
    print(model.summary())
    return model


def get_multi_reso_for_finger(img_rows,img_cols,num_kern,num_f1,num_f2,cnn_out_dim):
    padding='valid'
    activation='relu'
    inputs_r0 = Input((img_rows, img_cols, 1),name='input0')
    conv1 = Conv2D(num_kern[0], (5,5), activation=activation, padding=padding)(inputs_r0)
    pool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

    conv2 = Conv2D(num_kern[1], (3,3), activation=activation, padding=padding)(pool1)
    conv_r0 = MaxPooling2D(pool_size=(2, 2))(conv2)

    inputs_r1 = Input((img_rows/2, img_cols/2, 1),name='input1')
    conv1 = Conv2D(num_kern[0], (3,3), activation=activation, padding=padding)(inputs_r1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_kern[1], (3,3), activation=activation, padding=padding)(pool1)
    conv_r1 = MaxPooling2D(pool_size=(2, 2))(conv2)


    batch_norm = concatenate([conv_r0,conv_r1], axis=3)
    batch_norm=BatchNormalization()(batch_norm)

    convout = Flatten()(batch_norm)


    fullconnet1 = Dense(num_f1,activation=activation)(convout)
    fullconnet1 = Dropout(0.3)(fullconnet1)
    fullconnet11 = Dense(num_f2,activation=activation)(fullconnet1)
    fullconnet11 = Dropout(0.3)(fullconnet11)
    reg_out = Dense(cnn_out_dim,activation=None)(fullconnet11)
    model = Model(inputs=[inputs_r0,inputs_r1], outputs=[reg_out])
    print(model.summary())
    return model

