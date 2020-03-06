from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D,BatchNormalization,Dense,Flatten,Dropout
from keras.optimizers import Adam
import keras.backend as K
import constants

def multireso_net():
  num_filter = [64, 96, 128]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    x = _conv_maxpool(x, num_filter[2], k[2], p[2])
    return x

  input_image_0 = Input((constants.IMG_RSZ, constants.IMG_RSZ, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 4, 3], [4, 2, 2])

  input_image_1 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [5, 3, 3], [2, 2, 2])

  input_image_2 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_2 = _conv_block(input_image_2, num_filter, [5, 5, 3], [2, 1, 1])

  concat_layer = concatenate([conv_block_0,conv_block_1,conv_block_2], axis=3)
  batch_norm = BatchNormalization()(concat_layer)
  conv_out = Flatten()(batch_norm)

  conv_model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=conv_out)
  num_fc_filter = int(conv_model.output_shape[1])

  full_con_0 = Dense(num_fc_filter//2,activation='relu')(conv_out)
  full_con_0 = Dropout(0.5)(full_con_0)
  full_con_1 = Dense(num_fc_filter//4,activation='relu')(full_con_0)
  full_con_1 = Dropout(0.5)(full_con_1)
  reg_out = Dense(63,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_no_dropout():
  num_filter = [64, 96, 128]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    x = _conv_maxpool(x, num_filter[2], k[2], p[2])
    return x

  input_image_0 = Input((constants.IMG_RSZ, constants.IMG_RSZ, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 4, 3], [4, 2, 2])

  input_image_1 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [5, 3, 3], [2, 2, 2])

  input_image_2 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_2 = _conv_block(input_image_2, num_filter, [5, 5, 3], [2, 1, 1])

  concat_layer = concatenate([conv_block_0,conv_block_1,conv_block_2], axis=3)
  batch_norm = BatchNormalization()(concat_layer)
  conv_out = Flatten()(batch_norm)

  conv_model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=conv_out)
  num_fc_filter = int(conv_model.output_shape[1])

  full_con_0 = Dense(num_fc_filter//2,activation='relu')(conv_out)
  full_con_1 = Dense(num_fc_filter//4,activation='relu')(full_con_0)
  reg_out = Dense(63,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_no_dropout_no_batchnorm_lr0003():
  num_filter = [64, 96, 128]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    x = _conv_maxpool(x, num_filter[2], k[2], p[2])
    return x

  input_image_0 = Input((constants.IMG_RSZ, constants.IMG_RSZ, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 4, 3], [4, 2, 2])

  input_image_1 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [5, 3, 3], [2, 2, 2])

  input_image_2 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_2 = _conv_block(input_image_2, num_filter, [5, 5, 3], [2, 1, 1])

  concat_layer = concatenate([conv_block_0,conv_block_1,conv_block_2], axis=3)
  conv_out = Flatten()(concat_layer)

  conv_model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=conv_out)
  num_fc_filter = int(conv_model.output_shape[1])

  full_con_0 = Dense(num_fc_filter//2,activation='relu')(conv_out)
  full_con_1 = Dense(num_fc_filter//4,activation='relu')(full_con_0)
  reg_out = Dense(63,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0003),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_no_dropout_no_batchnorm_lr001():
  num_filter = [64, 96, 128]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    x = _conv_maxpool(x, num_filter[2], k[2], p[2])
    return x

  input_image_0 = Input((constants.IMG_RSZ, constants.IMG_RSZ, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 4, 3], [4, 2, 2])

  input_image_1 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [5, 3, 3], [2, 2, 2])

  input_image_2 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_2 = _conv_block(input_image_2, num_filter, [5, 5, 3], [2, 1, 1])

  concat_layer = concatenate([conv_block_0,conv_block_1,conv_block_2], axis=3)
  conv_out = Flatten()(concat_layer)

  conv_model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=conv_out)
  num_fc_filter = int(conv_model.output_shape[1])

  full_con_0 = Dense(num_fc_filter//2,activation='relu')(conv_out)
  full_con_1 = Dense(num_fc_filter//4,activation='relu')(full_con_0)
  reg_out = Dense(63,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.001),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_no_dropout_no_batchnorm():
  num_filter = [64, 96, 128]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    x = _conv_maxpool(x, num_filter[2], k[2], p[2])
    return x

  input_image_0 = Input((constants.IMG_RSZ, constants.IMG_RSZ, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 4, 3], [4, 2, 2])

  input_image_1 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [5, 3, 3], [2, 2, 2])

  input_image_2 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_2 = _conv_block(input_image_2, num_filter, [5, 5, 3], [2, 1, 1])

  concat_layer = concatenate([conv_block_0,conv_block_1,conv_block_2], axis=3)
  conv_out = Flatten()(concat_layer)

  conv_model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=conv_out)
  num_fc_filter = int(conv_model.output_shape[1])

  full_con_0 = Dense(num_fc_filter//2,activation='relu')(conv_out)
  full_con_1 = Dense(num_fc_filter//4,activation='relu')(full_con_0)
  reg_out = Dense(63,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_dense_2048_1024():
  num_filter = [64, 96, 128]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    x = _conv_maxpool(x, num_filter[2], k[2], p[2])
    return x

  input_image_0 = Input((constants.IMG_RSZ, constants.IMG_RSZ, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 4, 3], [4, 2, 2])

  input_image_1 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [5, 3, 3], [2, 2, 2])

  input_image_2 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_2 = _conv_block(input_image_2, num_filter, [5, 5, 3], [2, 1, 1])

  concat_layer = concatenate([conv_block_0,conv_block_1,conv_block_2], axis=3)
  batch_norm = BatchNormalization()(concat_layer)
  conv_out = Flatten()(batch_norm)

  conv_model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=conv_out)
  num_fc_filter = int(conv_model.output_shape[1])

  full_con_0 = Dense(2048,activation='relu')(conv_out)
  full_con_1 = Dense(1024,activation='relu')(full_con_0)
  reg_out = Dense(63,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_shanxin():
  num_filter = [64, 96, 128]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    x = _conv_maxpool(x, num_filter[2], k[2], p[2])
    return x

  input_image_0 = Input((constants.IMG_RSZ, constants.IMG_RSZ, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 4, 3], [4, 2, 2])

  input_image_1 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [5, 3, 3], [2, 2, 2])

  input_image_2 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_2 = _conv_block(input_image_2, num_filter, [5, 5, 3], [2, 1, 1])

  concat_layer = concatenate([conv_block_0,conv_block_1,conv_block_2], axis=3)
  batch_norm = BatchNormalization()(concat_layer)
  conv_out = Flatten()(batch_norm)

  conv_model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=conv_out)
  num_fc_filter = int(conv_model.output_shape[1])

  full_con_0 = Dense(num_fc_filter//1,activation='relu')(conv_out)
  full_con_1 = Dense(num_fc_filter//2,activation='relu')(full_con_0)
  reg_out = Dense(63,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_palm():
  num_filter = [32, 64, 96]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    x = _conv_maxpool(x, num_filter[2], k[2], p[2])
    return x

  input_image_0 = Input((constants.IMG_RSZ, constants.IMG_RSZ, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 3, 3], [4, 2, 2])

  input_image_1 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [5, 3, 3], [2, 2, 2])

  input_image_2 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_2 = _conv_block(input_image_2, num_filter, [5, 5, 3], [2, 1, 1])

  concat_layer = concatenate([conv_block_0,conv_block_1,conv_block_2], axis=3)
  batch_norm = BatchNormalization()(concat_layer)
  conv_out = Flatten()(batch_norm)

  full_con_0 = Dense(2048,activation='relu')(conv_out)
  full_con_0 = Dropout(0.3)(full_con_0)
  full_con_1 = Dense(1024,activation='relu')(full_con_0)
  full_con_1 = Dropout(0.3)(full_con_1)
  reg_out = Dense(18,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1,input_image_2], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_pip():
  num_filter = [48, 96]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    return x

  input_image_0 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 3], [4, 2])

  input_image_1 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [3, 3], [2, 2])

  concat_layer = concatenate([conv_block_0,conv_block_1], axis=3)
  batch_norm = BatchNormalization()(concat_layer)
  conv_out = Flatten()(batch_norm)

  full_con_0 = Dense(1024,activation='relu')(conv_out)
  full_con_0 = Dropout(0.3)(full_con_0)
  full_con_1 = Dense(1024,activation='relu')(full_con_0)
  full_con_1 = Dropout(0.3)(full_con_1)
  reg_out = Dense(3,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')

  model.summary()

  return model

def multireso_net_dtip():
  num_filter = [48, 96]

  def _conv_maxpool(input_tensor, num_filter, k, p):
      padding = 'valid'
      activation = 'relu'
      x = Conv2D(num_filter, (k,k), activation=activation, padding=padding)(input_tensor)
      x = MaxPooling2D(pool_size=(p, p))(x)
      return x

  def _conv_block(input_image, num_filter, k, p):
    x = _conv_maxpool(input_image, num_filter[0], k[0], p[0])
    x = _conv_maxpool(x, num_filter[1], k[1], p[1])
    return x

  input_image_0 = Input((constants.IMG_RSZ/2, constants.IMG_RSZ/2, constants.CHANNEL))
  conv_block_0 = _conv_block(input_image_0, num_filter, [5, 3], [4, 2])

  input_image_1 = Input((constants.IMG_RSZ/4, constants.IMG_RSZ/4, constants.CHANNEL))
  conv_block_1 = _conv_block(input_image_1, num_filter, [3, 3], [2, 2])

  concat_layer = concatenate([conv_block_0,conv_block_1], axis=3)
  batch_norm = BatchNormalization()(concat_layer)
  conv_out = Flatten()(batch_norm)

  full_con_0 = Dense(1024,activation='relu')(conv_out)
  full_con_0 = Dropout(0.3)(full_con_0)
  full_con_1 = Dense(1024,activation='relu')(full_con_0)
  full_con_1 = Dropout(0.3)(full_con_1)
  reg_out = Dense(6,activation=None)(full_con_1)
  model = Model(inputs=[input_image_0,input_image_1], outputs=reg_out)
  model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')

  model.summary()

  return model
