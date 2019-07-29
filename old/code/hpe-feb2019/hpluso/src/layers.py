from keras import backend as K
from keras.layers import Layer, Conv2D, MaxPooling2D, Lambda, concatenate
import tensorflow as tf

import constants

def _append_layer(all_layers, layer):
  if all_layers:
    return all_layers.append(layer)

def conv(input_tensor, num_filters, k=3, all_layers=None, name=None):
    layer = Conv2D(filters=num_filters, kernel_size=(k, k), strides=(1, 1), padding='same')(input_tensor)
    _append_layer(all_layers, layer)
    return layer

def max_pool(input_tensor, all_layers=None):
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
    _append_layer(all_layers, layer)
    return layer

def route(idx, all_layers=None):
    layers = [all_layers[i] for i in idx]
    if len(layers) > 1:
        layer = concatenate(layers, name='route')
    else:
        layer = layers[0]

    _append_layer(all_layers, layer)
    return layer

def reorg(input_tensor, all_layers=None):
    input_shape = K.shape(input_tensor)

    output_shape = input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 * input_shape[3]
    def space_to_depth_x2(x):
        # Import currently required to make Lambda work.
        # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
        import tensorflow as tf
        return tf.space_to_depth(x, block_size=2)

    def space_to_depth_x2_output_shape(input_shape):
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 * input_shape[3]) if input_shape[1] else (input_shape[0], None, None, 4 * input_shape[3])

    layer = Lambda(space_to_depth_x2, output_shape=space_to_depth_x2_output_shape, name='reorg')(input_tensor)

    _append_layer(all_layers, layer)
    return layer

def offset_to_uvz(input_tensor):
    # layer to reconstruct uvz hpe for loss calculation
  def _offset_to_uvz(x):
    # input_shape: (batch_size, 13, 13, 320)
    # output_shape: (batch_size, 845, 63+1)

    C_u = constants.C_u
    C_v = constants.C_v
    C_z = constants.C_z

    input_shape = K.shape(x)
    input_grid = K.reshape(x, (input_shape[0], 845, 64))

    hpe_ofs = K.reshape(input_grid[:, :, :63], (input_shape[0], 845, 21, 3))
    conf = input_grid[:, :, 63]

    index = K.arange(845)

    # (u,v,z) from index
    unravel_index = K.cast(K.transpose(tf.unravel_index(index, (13, 13, 5))), tf.float32)

    hpe_ofs_root = hpe_ofs[:, :, 0, :]
    hpe_ofs_rest = hpe_ofs[:, :, 1:, :] # bs, 845, 20, 3

    zero = K.zeros(K.shape(unravel_index[:, 0]))

    # w(x) = sigmoid(offset_x) + x (x= u,v,z) for hand root

    # [sigmoid(offset_u), 0, 0]
    u_sigmoid = tf.broadcast_to(tf.stack([K.sigmoid(unravel_index[:, 0]), zero, zero], axis=1), K.shape(hpe_ofs_root))
    # [ 0, [sigmoid(offset_v), 0]
    v_sigmoid = tf.broadcast_to(tf.stack([zero, K.sigmoid(unravel_index[:, 1]) , zero], axis=1), K.shape(hpe_ofs_root))
    z_sigmoid = tf.broadcast_to(tf.stack([K.sigmoid(unravel_index[:, 2]), zero, zero], axis=1), K.shape(hpe_ofs_root))

    hpe_uv_root =  hpe_ofs_root + C_u*u_sigmoid + C_v*v_sigmoid + C_z*z_sigmoid

    # w(x) = offset_x + x (x= u,v,z) for rest
    u = tf.broadcast_to(tf.expand_dims(tf.stack([unravel_index[:, 0], zero, zero], axis=1), axis=1), K.shape(hpe_ofs_rest))
    v = tf.broadcast_to(tf.expand_dims(tf.stack([zero, unravel_index[:, 1] , zero], axis=1), axis=1), K.shape(hpe_ofs_rest))
    z = tf.broadcast_to(tf.expand_dims(tf.stack([unravel_index[:, 2], zero, zero], axis=1), axis=1), K.shape(hpe_ofs_rest))

    hpe_uv_rest = hpe_ofs_rest + C_u*u + C_v*v + C_z*z

    hpe_uv = K.reshape(tf.concat([tf.expand_dims(hpe_uv_root, axis=2), hpe_uv_rest], axis=2), (input_shape[0], 845, 63) )

    output = tf.concat([hpe_uv, tf.expand_dims(conf, axis=-1)], axis=-1)

    return output

  return Lambda(_offset_to_uvz)(input_tensor)

def reshape(input_tensor):
    def _reshape(x):
        input_shape = K.shape(x)
        return(K.reshape(x, (input_shape[0], 845, 63)))
    return Lambda(_reshape)(input_tensor)
