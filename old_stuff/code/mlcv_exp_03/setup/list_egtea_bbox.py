import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_egtea_bbox():
    img_root = Path(ROOT)/'datasets'/'EGTEA'

    train_img_list  = []
    train_bbox_list  = []
    val_img_list    = []
    val_bbox_list    = []
    val_len = 400
    pad = 10
    img_path = img_root/'Images'
    img_list = [str(x) for x in sorted(img_path.glob('*')) if x.is_file()]

    val_index = np.arange(0, len(img_list), len(img_list)//val_len)
    for i, img_file in enumerate(img_list):
        print('{}/{}'.format(i, len(img_list)))
        if i in val_index:
            mask_path = img_file.replace('.jpg', '.png').replace('Images', 'Masks')
            mask = cv2.imread(mask_path, 0)
            large_comp, small_comp = seperate_2_blobs_in_mask(mask)

            bbox = get_bbox_from_mask(large_comp, pad=pad)
            if np.sum(bbox) != 0:
                val_img_list.append(img_file.split('/')[-1])
                val_bbox_list.append([float(x) for x in bbox])

            bbox = get_bbox_from_mask(small_comp, pad=pad)
            if np.sum(bbox) != 0:
                val_img_list.append(img_file.split('/')[-1])
                val_bbox_list.append([float(x) for x in bbox])
        else:
            mask_path = img_file.replace('.jpg', '.png').replace('Images', 'Masks')
            mask = cv2.imread(mask_path, 0)
            large_comp, small_comp = seperate_2_blobs_in_mask(mask)

            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(large_comp)
            # ax[1].imshow(small_comp)
            # ax[2].imshow(mask)
            # plt.show()

            bbox = get_bbox_from_mask(large_comp, pad=pad)
            if np.sum(bbox) != 0:
                train_img_list.append(img_file.split('/')[-1])
                train_bbox_list.append([float(x) for x in bbox])

            bbox = get_bbox_from_mask(small_comp, pad=pad)
            if np.sum(bbox) != 0:
                train_img_list.append(img_file.split('/')[-1])
                train_bbox_list.append([float(x) for x in bbox])

    parent_dir = Path(__file__).absolute().parents[1]
    np.savetxt(parent_dir/'data'/'labels'/'egtea_bbox_pad{}_train.txt'.format(pad), train_bbox_list)
    np.savetxt(parent_dir/'data'/'labels'/'egtea_bbox_pad{}_val.txt'.format(pad), val_bbox_list)
    with open(parent_dir/'data'/'labels'/'egtea_img_train.txt', 'w') as f:
        for i in train_img_list:
            f.write("%s\n" %i)
    with open(parent_dir/'data'/'labels'/'egtea_img_val.txt', 'w') as f:
        for i in val_img_list:
            f.write("%s\n" %i)

if __name__ == '__main__':
    list_egtea_bbox()