import numpy as np
import torchvision
import cv2

from src.datasets.transforms import *

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

def get_norm_bbox(uvd_gt, img_size, pad=50):
    """
    Get normalized bounding box for hand
    Args:
        uvd_gt          : Hand points in img space (-1, 2++)
    Out:
        bbox            : [x_cen ,y_cen, w, h]
    """
    x_max = int(np.amax(uvd_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt[:,1])) - pad, 0)

    # ensure bbox is within img bounds
    if y_max > img_size[1]:
        y_max = img_size[1]
    if y_min < 0:
        y_min = 0
    if x_max > img_size[0]:
        x_max = img_size[0]
    if x_min < 0:
        x_min = 0

    width   = (x_max - x_min)/img_size[0]
    height  = (y_max - y_min)/img_size[1]
    x_cen   = ((x_min + x_max)/2)/img_size[0]
    y_cen   = ((y_min + y_max)/2)/img_size[1]

    return np.asarray([x_cen, y_cen, width, height])