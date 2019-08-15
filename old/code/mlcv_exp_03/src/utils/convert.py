import numpy as np

def xyxy2xywh(x):
    """" convert box type xyxy to xywh """
    y = np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2])/2
    y[..., 1] = (x[..., 1] + x[..., 3])/2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def xywh2xyxy(x):
    """" convert box type xywh to xyxy """
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2]/2
    y[..., 1] = x[..., 1] - x[..., 3]/2
    y[..., 2] = x[..., 0] + x[..., 2]/2
    y[..., 3] = x[..., 1] + x[..., 3]/2
    return y