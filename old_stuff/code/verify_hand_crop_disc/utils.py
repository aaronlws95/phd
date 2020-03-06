import numpy as np
import random
from pathlib import Path

with open(Path(__file__).absolute().parents[0]/'data/root.txt') as f:
    ROOT = f.readlines()[0]

def parse(path):
    """ Parse the cfg file """
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    # Get rid of whitespaces
    lines = [x.rstrip().lstrip() for x in lines]
    data_cfg = {}
    for line in lines:
        key, value = line.split('=')
        value = value.strip()
        value = None if value.lower() == 'false' else value
        data_cfg[key.rstrip()] = value
    return data_cfg

def get_img_crop(uvd_gt, img, iou_thresh, pad=50):
    img_size = img.shape

    x_max = int(np.amax(uvd_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt[:,1])) - pad, 0)
    # ensure bbox is within img bounds
    if y_max > img_size[0]:
        y_max = img_size[0]
    if y_min < 0:
        y_min = 0
    if x_max > img_size[1]:
        x_max = img_size[1]
    if x_min < 0:
        x_min = 0

    width = x_max - x_min
    height = y_max - y_min

#     x_noise = random.uniform(-jitter, jitter)*width
#     y_noise = random.uniform(-jitter, jitter)*height
    x_noise = 0
    y_noise = 0
    y_min_noise = int(y_min + y_noise)
    y_max_noise = int(y_max + y_noise)
    x_min_noise = int(x_min + x_noise)
    x_max_noise = int(x_max + x_noise)

    if y_max_noise > img_size[0]:
        y_max_noise = img_size[0]
    if y_min_noise < 0:
        y_min_noise = 0
    if x_max_noise > img_size[1]:
        x_max_noise = img_size[1]
    if x_min_noise < 0:
        x_min_noise = 0

    hand_crop = img[y_min_noise:y_max_noise, x_min_noise:x_max_noise, :].copy()
    hand_box = np.asarray([(x_min_noise+x_max_noise)/2, (y_min_noise+y_max_noise)/2, width, height])

    if random.random() < 0.3:
        return hand_crop, 1

    def get_random_crop():
        rand_x = random.randint(0, img_size[1]-width)
        rand_y = random.randint(0, img_size[0]-height)
        no_hand_crop = img[rand_y:rand_y+height, rand_x:rand_x+width, :].copy()
        no_hand_box = np.asarray([(rand_x+rand_x+width)/2, (rand_y+rand_y+height)/2, width, height])
        return no_hand_crop, no_hand_box

    img_crop, img_box = get_random_crop()
    iou = bbox_iou(hand_box, img_box)
    if iou > iou_thresh:
        return img_crop, 1
    else:
        return img_crop, 0

def bbox_iou(box1, box2):
    """
    Calculate bounding box iou
    Args:
        box: [x_cen, y_cen, w, h] (4)
    Out:
        iou: (1)
    """
    mx      = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
    Mx      = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
    my      = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
    My      = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
    w1      = box1[2]
    h1      = box1[3]
    w2      = box2[2]
    h2      = box2[3]
    uw      = Mx - mx
    uh      = My - my
    cw      = w1 + w2 - uw
    ch      = h1 + h2 - uh
    carea   = 0

    if cw <= 0 or ch <= 0:
        return 0.0

    area1   = w1*h1
    area2   = w2*h2
    carea   = cw*ch
    uarea   = area1 + area2 - carea
    return carea/uarea