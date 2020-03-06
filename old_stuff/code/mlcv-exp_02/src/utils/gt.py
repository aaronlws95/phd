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

def create_multiple_gaussian_map(uvd_gt, output_size, sigma=25.0):
    coords_uv = np.stack([uvd_gt[:, 1], uvd_gt[:, 0]], -1)

    coords_uv = coords_uv.astype(np.int32)

    cond_1_in = np.logical_and(np.less(coords_uv[:, 0], output_size[0]-1), np.greater(coords_uv[:, 0], 0))
    cond_2_in = np.logical_and(np.less(coords_uv[:, 1], output_size[1]-1), np.greater(coords_uv[:, 1], 0))
    cond = np.logical_and(cond_1_in, cond_2_in)

    coords_uv = coords_uv.astype(np.float32)

    # create meshgrid
    x_range = np.expand_dims(np.arange(output_size[0]), 1)
    y_range = np.expand_dims(np.arange(output_size[1]), 0)

    X = np.tile(x_range, [1, output_size[1]]).astype(np.float32)
    Y = np.tile(y_range, [output_size[0], 1]).astype(np.float32)

    X = np.expand_dims(X, -1)
    Y = np.expand_dims(Y, -1)

    X_b = np.tile(X, [1, 1, coords_uv.shape[0]])
    Y_b = np.tile(Y, [1, 1, coords_uv.shape[0]])

    X_b -= coords_uv[:, 0]
    Y_b -= coords_uv[:, 1]

    dist = np.square(X_b) + np.square(Y_b)

    scoremap = np.exp(-dist / np.square(sigma)) * cond.astype(np.float32)

    return scoremap