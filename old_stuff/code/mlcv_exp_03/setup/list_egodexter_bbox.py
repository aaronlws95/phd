import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_egodexter_bbox():
    """ Prepare list for FPHA dataset """
    img_root    = Path(ROOT)/'datasets'/'EgoDexter'
    img_size    = (640, 480)
    pad         = 50

    # mode = '' #all
    # mode = '_five' # five finger present
    mode = '_five_nokitchen' # kitchen has bad annot
    # Accumulate list
    train_img_list  = []
    val_img_list   = []
    train_bbox_list = []
    val_bbox_list  = []

    data_path = img_root/'data'
    # sequences = ['Desk', 'Fruits', 'Kitchen', 'Rotunda']
    sequences = ['Desk', 'Fruits', 'Rotunda']

    for seq in sequences:
        cur_data_path = data_path/seq
        cur_img_path = cur_data_path/'color'
        img_list = [str(x) for x in sorted(cur_img_path.glob('*')) if x.is_file()]
        with open(cur_data_path/'annotation.txt_3D.txt') as f:
            skel_list = f.readlines()

        for i, (img_path, skel) in enumerate(zip(img_list, skel_list)):
            new_skel = re.split('; |,', skel)[:-1]
            new_skel = np.asarray(new_skel).reshape(-1, 3).astype(np.float32)
            new_skel = new_skel[~np.all(new_skel == 0, axis=1)]
            if not ('_five' in mode and len(new_skel) != 5):
                if np.sum(new_skel) != 0:
                    new_skel_uvd = EGOD.xyz2uvd_color(new_skel)
                    bbox = get_bbox_from_pose(new_skel_uvd, img_size=img_size, pad=pad)

                    path_split = img_path.split('/')
                    if i%100 != 0:
                        train_img_list.append('{}/{}/{}'.format(path_split[-3], path_split[-2], path_split[-1]))
                        train_bbox_list.append(bbox)
                    else:
                        val_img_list.append('{}/{}/{}'.format(path_split[-3], path_split[-2], path_split[-1]))
                        val_bbox_list.append(bbox)

                    # Display
                    # fig, ax = plt.subplots()
                    # img_show = cv2.imread(img_path)
                    # ax.imshow(img_show)
                    # draw_bbox(ax, bbox)
                    # print(new_skel_uvd)
                    # ax.scatter(new_skel_uvd[..., 0], new_skel_uvd[..., 1], c='r')
                    # plt.show()

    # Save
    parent_dir = Path(__file__).absolute().parents[1]
    np.savetxt(parent_dir/'data'/'labels'/'egodexter_bbox_pad{}{}_train.txt'.format(pad, mode), train_bbox_list)
    np.savetxt(parent_dir/'data'/'labels'/'egodexter_bbox_pad{}{}_val.txt'.format(pad, mode), val_bbox_list)
    with open(parent_dir/'data'/'labels'/'egodexter_img{}_train.txt'.format(mode), 'w') as f:
        for i in train_img_list:
            f.write("%s\n" %i)
    with open(parent_dir/'data'/'labels'/'egodexter_img{}_val.txt'.format(mode), 'w') as f:
        for i in val_img_list:
            f.write("%s\n" %i)

if __name__ == '__main__':
    list_egodexter_bbox()