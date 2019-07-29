import random
import math
import cv2
import numpy    as np
from PIL        import Image
        
# ========================================================
# UTILITIES
# ========================================================

def imgshape2torch(img):
    """ (H, W, C) to (C, H, W) supports batch """
    img = np.swapaxes(img, -3, -1).astype("float32")
    return np.swapaxes(img, -2, -1).astype("float32")

def torchshape2img(img):
    """ (C, H, W) to (H, W, C) supports batch """
    img = np.swapaxes(img, -2, -1).astype("float32")
    return np.swapaxes(img, -3, -1).astype("float32")

def scale_img_255(img):
    """ Scale image to range 0-255 """
    return cv2.normalize(np.asarray(img), None, alpha = 0, beta = 255,
                         norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC3)

def resize_img(img, new_size):
    """ Resize image """
    img_resize = cv2.resize(img, dsize=new_size,
                            interpolation=cv2.INTER_CUBIC)
    return img_resize

def scale_points_WH(points, i_shape, o_shape):
    """
    Scale points to new size
    Args:
        points      : (b, -1, 2++)
        i_shape     : input shape (x, y)
        o_shape     : output_shape (x, y)
    Out:
        new_points  :  Scaled points
    """
    new_points          = points.copy()
    new_points[..., 0] *= o_shape[0]/i_shape[0]
    new_points[..., 1] *= o_shape[1]/i_shape[1]
    return new_points

# ========================================================
# AUGMENT
# ========================================================

def rotate_points(pts, degrees, cx, cy, w, h):
    """ Rotate points
    Args:
        pts         : Points to rotate (-1, 2)
        degrees     : Angle in degrees
        cx          : Image x centre
        cy          : Image y centre
        w           : Image width
        h           : Image height
    Out:
        calculated  : Rotated angle
    """
    pts         = np.hstack((pts, np.ones((pts.shape[0], 1))))

    M           = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)

    cos         = np.abs(M[0, 0])
    sin         = np.abs(M[0, 1])

    nW          = int((h * sin) + (w * cos))
    nH          = int((h * cos) + (w * sin))

    calculated  = np.dot(M, pts.T).T
    return calculated

def get_corners(bboxes):
    """ Get corners of bounding boxes
    Args:
        bboxes: Numpy array containing bounding boxes of shape `N X 4` where N
                is the number of bounding boxes and the bounding boxes are
                represented in the format `x1 y1 x2 y2`
    Out:
        corners: Numpy array of shape `N x 8` containing N bounding boxes each
                 described by their corner co-ordinates
                 `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    width   = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height  = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)

    x1      = bboxes[:,0].reshape(-1,1)
    y1      = bboxes[:,1].reshape(-1,1)

    x2      = x1 + width
    y2      = y1

    x3      = x1
    y3      = y1 + height

    x4      = bboxes[:,2].reshape(-1,1)
    y4      = bboxes[:,3].reshape(-1,1)

    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    return corners

def rotate_box(corners, degrees, cx, cy, w, h):
    """ Rotate the bounding box.
    Args:
        corners : Numpy array of shape `N x 8` containing N bounding boxes
                  each described by their corner co-ordinates
                  `x1 y1 x2 y2 x3 y3 x4 y4`
        degrees : angle by which the image is to be rotated in degrees
        cx      : x coordinate of the center of image
                  (about which the box will be rotated)
        cy      : y coordinate of the center of image
                  (about which the box will be rotated)
        h       : height of the image
        w       : width of the image
    Out:
        calc    : Numpy array of shape `N x 8` containing N rotated bounding
                  boxes each described by their corner co-ordinates
                  `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1),
                                          dtype = type(corners[0][0]))))

    M       = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)

    cos     = np.abs(M[0, 0])
    sin     = np.abs(M[0, 1])

    nW      = int((h * sin) + (w * cos))
    nH      = int((h * cos) + (w * sin))

    # Prepare the vector to be transformed
    calc    = np.dot(M,corners.T).T
    calc    = calc.reshape(-1,8)
    return calc

def get_enclosing_box(corners):
    """ Get an enclosing box for rotated corners of a bounding box
    Args:
        corners : Numpy array of shape `N x 8` containing N bounding boxes
                  each described by their corner co-ordinates
                  `x1 y1 x2 y2 x3 y3 x4 y4`
    Out:
        final   : Numpy array containing enclosing bounding boxes of shape
                  `N X 4` where N is the number of bounding boxes and the
                  bounding boxes are represented in the format `x1 y1 x2 y2`
    """
    x_      = corners[:,[0,2,4,6]]
    y_      = corners[:,[1,3,5,7]]

    xmin    = np.min(x_,1).reshape(-1,1)
    ymin    = np.min(y_,1).reshape(-1,1)
    xmax    = np.max(x_,1).reshape(-1,1)
    ymax    = np.max(y_,1).reshape(-1,1)

    final   = np.hstack((xmin, ymin, xmax, ymax))
    return final

def update_rotate_bbox(bboxes, degrees, cx, cy, w, h):
    """ Rotate bbox points
    Args:
        bboxes  : [x_min, y_min, x_max, y_max] (4)
        degrees : angle in degrees
        cx      : x coordinate of the center of image
                  (about which the box will be rotated)
        cy      : y coordinate of the center of image
                  (about which the box will be rotated)
        h       : height of the image
        w       : width of the image
    Out:
        new_bbox: Bbox moved to new location
    """
    corners         = get_corners(bboxes)
    corners         = np.hstack((corners, bboxes[:, 4:]))
    corners[:, :8]  = rotate_box(corners[:, :8], degrees, cx, cy, w, h)
    new_bbox        = get_enclosing_box(corners)
    return new_bbox

def jitter_points(points, ofs_info):
    """
    Jitter points to match jittered image
    Args:
        points      : Points scaled to (1, 1) (-1, 2++)
        ofs_info    : Info from jitter_img
    Out:
        new_points  : Jittered points (-1, 2++)
    """
    sx, sy, dx, dy      = ofs_info
    new_points          = points.copy()
    new_points[:, 0]    = np.minimum(0.999,
                                     np.maximum(0, (new_points[:, 0] - dx)*sx))
    new_points[:, 1]    = np.minimum(0.999,
                                     np.maximum(0, (new_points[:, 1] - dy)*sy))
    return new_points

def aug_img_mask(img, mask, jitter, is_flip,
                 hue, sat, exp, crop, new_shape=None):
    """ Simultaneously augment image and mask """
    w, h = img.size

    # if cropping no need to jitter
    if new_shape and crop:
        pleft       = random.randint(0, w - new_shape[0])
        ptop        = random.randint(0, h - new_shape[1])
        new_img     = img.crop((pleft,
                                ptop,
                                pleft + new_shape[0],
                                ptop + new_shape[1]))
        new_mask    = mask.crop((pleft,
                                 ptop,
                                 pleft + new_shape[0],
                                 ptop + new_shape[1]))
    else:
        # Jitter
        dw =int(w*jitter)
        dh =int(h*jitter)

        pleft  = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop   = random.randint(-dh, dh)
        pbot   = random.randint(-dh, dh)

        swidth =  w - pleft - pright
        sheight = h - ptop - pbot

        sx =  w / float(swidth)
        sy =  h / float(sheight)

        new_img = img.crop((pleft,
                            ptop,
                            pleft + swidth - 1,
                            ptop + sheight - 1))
        new_mask = mask.crop((pleft,
                            ptop,
                            pleft + swidth - 1,
                            ptop + sheight - 1))
        # Resize
        if new_shape:
            new_img = new_img.resize(new_shape)
            new_mask = new_mask.resize(new_shape)
        else:
            new_img = new_img.resize((w, h))
            new_mask = new_mask.resize((w, h))

    # Flip
    if is_flip:
        flip = random.randint(1,10000)%2
        if flip:
            new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
            new_mask = new_mask.transpose(Image.FLIP_LEFT_RIGHT)

    # Distort HSV
    new_img = new_img.convert('HSV')
    cs = list(new_img.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * exp)
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    new_img = Image.merge(new_img.mode, tuple(cs))
    new_img = new_img.convert('RGB')
    return new_img, new_mask

def flip_img(img):
    """ Randomly flip image along the vertical axis """
    flip = random.randint(1, 10000)%2
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img, flip

def jitter_img(img, jitter, new_shape=None):
    """
    Randomly jitter the image
    Args:
        img                 : PIL Image
        jitter              : Amount to jitter
        new_shape           : New shape to resize
    Out:
        sized               : Jittered Image
        (sx, sy, dx, dy)    : Jittered information for points
    """
    w, h    = img.size

    dw      = int(w*jitter)
    dh      = int(h*jitter)

    pleft   = random.randint(-dw, dw)
    pright  = random.randint(-dw, dw)
    ptop    = random.randint(-dh, dh)
    pbot    = random.randint(-dh, dh)

    swidth  = w - pleft - pright
    sheight = h - ptop - pbot

    sx      = w/float(swidth)
    sy      = h/float(sheight)

    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    if new_shape:
        sized = cropped.resize(new_shape)
    else:
        sized = cropped.resize((w,h))

    dx = float(pleft)/w
    dy = float(ptop)/h
    return sized, (sx, sy, dx, dy)

def distort_image_HSV(img, hue, sat, exp, rand=True):
    """
    Distort HSV for image
    Args:
        img: PIL image
    Out:
        img: HSV distorted PIL image
    """
    if rand:
        def rand_scale(s):
            scale = random.uniform(1, s)
            if(random.randint(1,10000)%2):
                return scale
            return 1./scale

        hue = random.uniform(-hue, hue)
        sat = rand_scale(sat)
        exp = rand_scale(exp)

    img = img.convert('HSV')
    cs = list(img.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * exp)

    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    img = Image.merge(img.mode, tuple(cs))
    img = img.convert('RGB')
    return img