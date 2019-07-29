import keras.backend as K
import tensorflow as tf

import constants

def l2_dist(a, b):
    return K.sqrt(K.sum(K.square(a - b), axis=-1, keepdims=True))

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# shape
# y_true: (63 + 1) uvz space
# y_pred: (b, 845, 63 + 1) uvz space
def hpo_loss(y_true, y_pred):
    d_th = constants.d_th
    a = constants.a

    conf_pred = y_pred[:, :, 63]

    hpe_pred = y_pred[:, :, :63]
    hpe_true = y_true[:, :, :63]

    hpe_loss = l2_dist(hpe_pred, hpe_true)
    y_pred_21_3 = K.reshape(hpe_pred, (-1, 845, 21, 3))
    y_true_21_3 = K.reshape(hpe_true, (-1, 845, 21, 3))

    y_pred_uv = K.reshape(y_pred_21_3[..., :2], (-1, 845, 42))
    y_true_uv = K.reshape(y_true_21_3[..., :2], (-1, 845, 42))

    DT_uv = l2_dist(y_pred_uv[..., :2], y_true_uv[..., :2])
    conf_uv_true = K.zeros(K.shape(conf_pred))
    conf_uv_true = conf_uv_true + K.squeeze(K.cast(K.greater(DT_uv,d_th), dtype=tf.float32)*K.exp(a*(1-(DT_uv/d_th))), axis=-1)

    y_pred_z = K.reshape(y_pred_21_3[..., 2], (-1, 845, 21))
    y_true_z = K.reshape(y_true_21_3[..., 2], (-1, 845, 21))

    DT_z = l2_dist(y_pred_z[..., 2], y_true_z[..., 2])
    conf_z_true = K.zeros(K.shape(conf_pred))
    conf_z_true = conf_z_true + K.cast(K.greater(DT_z,d_th), dtype=tf.float32)*K.exp(a*(1-(DT_z/d_th)))

    conf_true = 0.5*conf_uv_true+ 0.5*conf_z_true
    conf_loss = l2_dist(K.expand_dims(conf_pred, axis=-1), K.expand_dims(conf_true, axis=-1))

    return hpe_loss + conf_loss

