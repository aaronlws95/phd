import os
import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw
from tqdm import tqdm
import h5py
import xml.etree.ElementTree as ET

PATH_TO_WORKDIR = '/media/aaron/DATA/ubuntu/hand-seg/hand_over_face'
PATH_TO_DATA = '/media/aaron/DATA/ubuntu/hand_over_face'
PATH_TO_IMG = os.path.join(PATH_TO_DATA, 'images_resized')
PATH_TO_MASK = os.path.join(PATH_TO_DATA, 'masks')
PATH_TO_ANNOT = os.path.join(PATH_TO_DATA, 'annotations')
MAX_ID = 302

def get_img(img_id, resize_dim=None, is_gray=False):
    img = cv2.imread(os.path.join(PATH_TO_IMG, '%d.jpg' %img_id))
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize_dim:
        img = cv2.resize(img, resize_dim)
    img = np.asarray(img, np.float32)
    return img

def get_mask(img_id, resize_dim=None):
    img = cv2.imread(os.path.join(PATH_TO_MASK, '%d.png' %img_id))
    if resize_dim:
        img = cv2.resize(img, resize_dim)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    return img

# def get_mask_pts(img_id):
#     e = ET.parse(os.path.join(PATH_TO_ANNOT, '%d.xml' %img_id))
#     e_root = e.getroot()
#     mask_pts = []
#     for elem in e_root.find('object/polygon'):
#         if elem.tag == 'pt':
#             x = int(elem.find('x').text)
#             y = int(elem.find('y').text)
#             mask_pts.append((x, y))
#     return mask_pts

def show_mask_and_img(img, mask):
    alpha = 0.5

    # if RGB
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    mask = cv2.normalize(mask, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    out_img = (1-alpha)*img + alpha*mask

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('img')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('mask')
    ax[2].imshow(out_img, cmap='gray')
    ax[2].set_title('img+mask')
    plt.show()

def save_mask_data_split(data_split):
    def get_mask_data_split(data_file, resize_dim=(256, 256)):
        mask = []
        with open(data_file) as f:
            img_list = f.readlines()
            for img_id in img_list:
                ref_img_id = int(img_id.strip())
                mask.append(get_mask(ref_img_id, resize_dim))
        mask = np.asarray(mask)
        return mask

    data_file = os.path.join(PATH_TO_WORKDIR, 'data_%s.txt' %data_split)
    save_path = os.path.join(PATH_TO_WORKDIR, 'mask_%s.h5' %data_split)
    print('saving mask to %s' %save_path)
    h5f = h5py.File(save_path, 'w')
    mask = get_mask_data_split(data_file)
    h5f.create_dataset('mask', data=mask)
    h5f.close()

# img_id = 1
# is_gray = True
# resize_dim = None #(384,216)

# for img_id in range(1, 302):
#     show_mask_and_img(get_img(img_id, resize_dim=resize_dim, is_gray=is_gray), get_mask(img_id, resize_dim=resize_dim))
