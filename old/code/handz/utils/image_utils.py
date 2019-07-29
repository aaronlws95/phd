import random
import numpy as np
import cv2
from PIL import Image
import math

# ========================================================
# IMG UTILS
# ========================================================

def imgshape2torch(img):
    img = np.swapaxes(img, 0, 2).astype("float32")
    return np.swapaxes(img, 1, 2).astype("float32")

def torchshape2img(img):
    img = np.swapaxes(img, 1, 2).astype("float32")
    return np.swapaxes(img, 0, 2).astype("float32")

def scale_img_255(img):
    return cv2.normalize(np.asarray(img), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC3)

def resize_img(img, new_size):
    from skimage.transform import resize
    img_resize = resize(img, new_size, order=3, preserve_range=True).astype('uint32')
    return img_resize

# ========================================================
# AUGMENT
# ========================================================

def scale_points_WH(points, i_shape, o_shape):
    """
    input: list of 2D points
    output: list of 2D points scaled to o_shape dim (W,H)
    """
    new_points = points.copy()
    new_points[..., 0] *= o_shape[0]/i_shape[0]
    new_points[..., 1] *= o_shape[1]/i_shape[1]
    return new_points

def jitter_points(points, ofs_info):
    """
    input: points (only care about 2D), ofs_info from jitter_img
    output: Jittered points
    """
    sx, sy, dx, dy = ofs_info
    new_points = points.copy()
    new_points[:, 0] = np.minimum(0.999, np.maximum(0, (new_points[:, 0] - dx)*sx))
    new_points[:, 1] = np.minimum(0.999, np.maximum(0, (new_points[:, 1] - dy)*sy))
    
    return new_points

def flip_img(img):
    flip = random.randint(1,10000)%2
    if flip: 
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img, flip

def jitter_img(img, jitter, new_shape=None):
    """
    input: PIL Image
    output: Jittered Image
    """
    w, h = img.size
    
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
    
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))
    
    if new_shape:
        sized = cropped.resize(new_shape)
    else:
        sized = cropped.resize((w,h))
    
    dx = float(pleft) / w
    dy = float(ptop) / h
    return sized, (sx, sy, dx, dy)

def distort_image_HSV(img, hue, sat, exp, rand=True):
    """
    input: PIL image
    output: HSV distorted PIL image
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

def letterbox(img, height=416, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def random_affine(img, targets, degrees, translate, scale, shear, borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction of bounding boxes
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        np.clip(xy, 0, height, out=xy)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets

def random_affine_pts(img, targets, degrees, translate, scale, shear, borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        points = np.ones((targets.shape[0], targets.shape[1]+1))
        points[:, :-1] = targets
        points = (points @ M.T)
    return imw, points[:, :2]