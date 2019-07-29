import keras.backend as K

# metrics
def dice_coef(y_true, y_pred, smooth=1e-7):
    intersection = y_true * y_pred
    sum_true_pred = K.sum(y_true) + K.sum(y_pred)
    dice = (2. * K.sum(intersection) + smooth) / (sum_true_pred + smooth)
    return dice

def jaccard_dist(y_true, y_pred, smooth=1e-7):
    intersection = y_true * y_pred
    sum_true_pred = y_true + y_pred
    union = sum_true_pred - intersection
    jac = (K.sum(intersection) + smooth) / (K.sum(union) + smooth)
    return jac

# loss
def dice_coef_loss(y_true, y_pred, smooth=1e-7):
    return 1 - dice_coef(y_true, y_pred, smooth)

def jaccard_dist_loss(y_true, y_pred, smooth=1e-7):
    return 1 - jaccard_distance(y_true, y_pred, smooth)

custom_obj = {'dice_coef': dice_coef,
             'dice_coef_loss': dice_coef_loss,
             'jaccard_dist': jaccard_dist,
             'jaccard_dist_loss': jaccard_dist_loss}
