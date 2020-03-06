import numpy as np

def xyxy2xywh(x):
    """"
    Convert box type xyxy to xywh
    Args:
        x: (-1, 4) [x_min, y_min, x_max, y_max]
    Out:
        y: (-1, 4) [x_cen, y_cen, w, h]
    """
    y = np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2])/2
    y[..., 1] = (x[..., 1] + x[..., 3])/2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def xywh2xyxy(x):
    """"
    Convert box type xywh to xyxy
    Args:
        x: (-1, 4) [x_cen, y_cen, w, h]
    Out:
        y: (-1, 4) [x_min, y_min, x_max, y_max]
    """
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2]/2
    y[..., 1] = x[..., 1] - x[..., 3]/2
    y[..., 2] = x[..., 0] + x[..., 2]/2
    y[..., 3] = x[..., 1] + x[..., 3]/2
    return y