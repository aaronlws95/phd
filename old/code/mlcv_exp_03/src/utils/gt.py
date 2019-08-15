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

def seperate_2_blobs_in_mask(img):
    large_comp = np.zeros_like(img)

    for val in np.unique(img)[1:]:
        mask = np.uint8(img == val)
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        large_comp[labels == largest_label] = val
    small_comp = img.copy()
    small_comp[np.where(large_comp != 0)] = 0

    return large_comp, small_comp