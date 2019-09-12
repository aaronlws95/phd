import numpy as np
import torchvision
import cv2

def get_img_dataloader(img_path, img_rsz):
    from src.datasets.transforms import ImgResize, ImgToTorch
    tfrm = []
    tfrm.append(ImgResize((img_rsz)))
    tfrm.append(ImgToTorch())
    transform = torchvision.transforms.Compose(tfrm)

    img         = cv2.imread(img_path)[:, :, ::-1]
    sample      = {'img': img}
    sample      = transform(sample)
    img         = sample['img']

    return img

def get_bbox_from_mask(mask, pad=10, mode='xywh'):
    mask_idx = np.where(mask != 0)
    if(mask_idx[0].size == 0 or mask_idx[1].size == 0):
        return np.asarray([0, 0, 0, 0])
    y_min = np.min(mask_idx[0]) - pad
    x_min = np.min(mask_idx[1]) - pad
    y_max = np.max(mask_idx[0]) + pad
    x_max = np.max(mask_idx[1]) + pad

    if mode == 'xywh':
        width   = (x_max - x_min)
        height  = (y_max - y_min)
        x_cen   = ((x_min + x_max)/2)
        y_cen   = ((y_min + y_max)/2)
        return np.asarray([x_cen, y_cen, width, height])
    elif mode == 'xyxy':
        return np.asarray([x_min, y_min, x_max, y_max])

def get_bbox_from_mask_pts(mask, pad=10, mode='xywh'):
    y_min = np.min(mask[..., 1]) - pad
    x_min = np.min(mask[..., 0]) - pad
    y_max = np.max(mask[..., 1]) + pad
    x_max = np.max(mask[..., 0]) + pad

    if mode == 'xywh':
        width   = (x_max - x_min)
        height  = (y_max - y_min)
        x_cen   = ((x_min + x_max)/2)
        y_cen   = ((y_min + y_max)/2)
        return np.asarray([x_cen, y_cen, width, height])
    elif mode == 'xyxy':
        return np.asarray([x_min, y_min, x_max, y_max])

def get_bbox_from_pose(pose, img_size=False, pad=50, mode='xywh', norm=False, within_img=True):
    """ get bbox given 2D pose """
    x_max = int(np.amax(pose[:,0])) + pad
    x_min = np.maximum(int(np.amin(pose[:,0])) - pad, 0)
    y_max = int(np.amax(pose[:,1])) + pad
    y_min = np.maximum(int(np.amin(pose[:,1])) - pad, 0)

    if img_size:
        img_width, img_height = img_size

    if within_img and img_size:
        if y_max > img_height:
            y_max = img_height
        if y_min < 0:
            y_min = 0
        if x_max > img_width:
            x_max = img_width
        if x_min < 0:
            x_min = 0

    if mode == 'xywh':
        width   = (x_max - x_min)
        height  = (y_max - y_min)
        x_cen   = ((x_min + x_max)/2)
        y_cen   = ((y_min + y_max)/2)
        if norm and img_size:
            width   = width/img_width
            height  = height/img_height
            x_cen   = x_cen/img_width
            y_cen   = y_cen/img_height
        return np.asarray([x_cen, y_cen, width, height])
    elif mode == 'xyxy':
        if norm and img_size:
            x_min = x_min/img_width
            y_min = y_min/img_height
            x_max = x_max/img_width
            y_max = y_max/img_height
        return np.asarray([x_min, y_min, x_max, y_max])
    else:
        raise ValueError('Invalid mode', mode)

def midpoint(p1, p2):
    return (p1 + p2)/2

def centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points), sum(z)/len(points))

    return centroid



def get_img_crop(uvd_gt, img, iou_thresh, pad=50, percent_true=0.3, jitter=0):
    import random
    from src.utils.yolo import bbox_iou

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

    if jitter != 0:
        x_noise = random.uniform(-jitter, jitter)*width
        y_noise = random.uniform(-jitter, jitter)*height
    else:
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

    ori_hand_box = np.asarray([(int(x_min)+int(x_max))/2, (int(y_min)+int(y_max))/2, width, height])

    iou = bbox_iou(hand_box, ori_hand_box)

    if random.random() < percent_true:
        return hand_crop, 1, iou

    def get_random_crop():
        rand_x = random.randint(0, img_size[1]-width)
        rand_y = random.randint(0, img_size[0]-height)
        no_hand_crop = img[rand_y:rand_y+height, rand_x:rand_x+width, :].copy()
        no_hand_box = np.asarray([(rand_x+rand_x+width)/2, (rand_y+rand_y+height)/2, width, height])
        return no_hand_crop, no_hand_box

    img_crop, img_box = get_random_crop()
    iou = bbox_iou(hand_box, img_box)
    if iou > iou_thresh:
        return img_crop, 1, iou
    else:
        return img_crop, 0, iou

def get_img_crop_from_bbox(img, bbox, mode='xywh', pad=0):
    from src.utils.convert import xywh2xyxy

    if mode == 'xywh':
        bbox = xywh2xyxy(bbox.copy())

    x_min, y_min, x_max, y_max = bbox

    if y_min < 0:
        y_min = 0
    if x_min < 0:
        x_min = 0
    if x_max > img.shape[1]:
        x_max = img.shape[1]
    if y_max > img.shape[0]:
        y_max = img.shape[0]

    crop = img[int(y_min-pad):int(y_max+pad), int(x_min-pad):int(x_max+pad), :].copy()

    return crop