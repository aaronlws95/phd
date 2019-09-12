import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_eyth_bbox():
    img_root = Path(ROOT)/'datasets'/'eyth_dataset'

    train_bbox_list  = []
    val_bbox_list    = []
    test_bbox_list   = []
    train_img_list   = []
    val_img_list     = []
    test_img_list    = []

    pad = 10
    img_path = img_root/'images'
    mask_path = img_root/'masks'

    split = ['test', 'train', 'val']

    for spl in split:
        with open(img_root/'train-val-test-split'/'{}.txt'.format(spl)) as f:
            lines = f.readlines()
            for l in tqdm(lines):
                mask_file = str(mask_path/l).strip()
                if l.split('/')[0] == 'vid6':
                    mask_file = mask_file.replace('.jpg', '.png')
                else:
                    mask_file = mask_file.replace('.jpg', '')
                mask = cv2.imread(mask_file, 0)
                large_comp, small_comp = seperate_2_blobs_in_mask(mask)

                bbox = get_bbox_from_mask(large_comp, pad=pad)
                bbox_large = bbox.copy()
                if np.sum(bbox) != 0:
                    if spl == 'test':
                        test_bbox_list.append(bbox)
                        test_img_list.append(l.strip())
                    elif spl == 'val':
                        val_bbox_list.append(bbox)
                        val_img_list.append(l.strip())
                    elif spl == 'train':
                        train_bbox_list.append(bbox)
                        train_img_list.append(l.strip())
                bbox = get_bbox_from_mask(small_comp, pad=pad)
                if np.sum(bbox) != 0:
                    if spl == 'test':
                        test_bbox_list.append(bbox)
                        test_img_list.append(l.strip())
                    elif spl == 'val':
                        val_bbox_list.append(bbox)
                        val_img_list.append(l.strip())
                    elif spl == 'train':
                        train_bbox_list.append(bbox)
                        train_img_list.append(l.strip())

                print(str(img_path/l).strip())
                fig, ax = plt.subplots(2, 2)
                img = cv2.imread(str(img_path/l).strip())[:, :, ::-1]
                ax[0, 0].imshow(large_comp)
                ax[0, 1].imshow(small_comp)
                ax[1, 0].imshow(mask)
                ax[1, 1].imshow(img)
                draw_bbox(ax[1, 1], bbox_large)
                draw_bbox(ax[1, 1], bbox)
                plt.show()

    # parent_dir = Path(__file__).absolute().parents[1]
    # np.savetxt(parent_dir/'data'/'labels'/'eyth_bbox_pad{}_train.txt'.format(pad), train_bbox_list)
    # np.savetxt(parent_dir/'data'/'labels'/'eyth_bbox_pad{}_val.txt'.format(pad), val_bbox_list)
    # np.savetxt(parent_dir/'data'/'labels'/'eyth_bbox_pad{}_test.txt'.format(pad), test_bbox_list)
    # with open(parent_dir/'data'/'labels'/'eyth_img_train.txt', 'w') as f:
    #     for i in train_img_list:
    #         f.write("%s\n" %i)
    # with open(parent_dir/'data'/'labels'/'eyth_img_val.txt', 'w') as f:
    #     for i in val_img_list:
    #         f.write("%s\n" %i)
    # with open(parent_dir/'data'/'labels'/'eyth_img_test.txt', 'w') as f:
    #     for i in test_img_list:
    #         f.write("%s\n" %i)

if __name__ == '__main__':
    list_eyth_bbox()