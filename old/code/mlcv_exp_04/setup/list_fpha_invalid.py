import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

parent_dir = Path(__file__).absolute().parents[1]
train_set   = parent_dir/'data'/'labels'/'fpha_img_train.txt'
test_set    = parent_dir/'data'/'labels'/'fpha_img_test.txt'
train_xyz   = parent_dir/'data'/'labels'/'fpha_xyz_train.txt'
test_xyz    = parent_dir/'data'/'labels'/'fpha_xyz_test.txt'
with open(train_set, 'r') as f:
    train_labels = f.read().splitlines()
with open(test_set, 'r') as f:
    test_labels = f.read().splitlines()
train_xyz_gt = np.reshape(np.loadtxt(train_xyz), (-1, 21, 3))
test_xyz_gt = np.reshape(np.loadtxt(test_xyz), (-1, 21, 3))
all_file_name = train_labels + test_labels
all_xyz_gt = np.concatenate((train_xyz_gt, test_xyz_gt))
all_uvd_gt = FPHA.xyz2uvd_color(all_xyz_gt)

def _get_all_vid_idx(img_list):
    all_vid_idx = []
    for subject_action_seq in img_list:
        for i, fn in enumerate(all_file_name):
            if subject_action_seq in fn:
                all_vid_idx.append(i)
    return all_vid_idx

def list_invalid_fpha():
    index0 = list(np.unique(np.argwhere(all_uvd_gt[..., 0] > FPHA.ORI_WIDTH)[:, 0]))
    index1 = list(np.unique(np.argwhere(all_uvd_gt[..., 0] < 0)[:, 0]))
    index2 = list(np.unique(np.argwhere(all_uvd_gt[..., 1] > FPHA.ORI_HEIGHT)[:, 0]))
    index3 = list(np.unique(np.argwhere(all_uvd_gt[..., 1] < 0)[:, 0]))

    outbound_index = np.unique(index0 + index1 + index2 + index3)

    bad_seqs = ['Subject_1/unfold_glasses/5',   # one finger is really long
                'Subject_4/read_letter/3',      # annotation gets stuck
                'Subject_5/use_flash/6',        # annotation completely wrong, no hand even in image
                'Subject_1/handshake/3',        # blurry image, wrong annotation
                'Subject_1/clean_glasses/2',    # bad annotation
                'Subject_1/clean_glasses/4',    # bad annotation
                'Subject_2/tear_paper/3',       # bad annotation
                'Subject_1/high_five/3',        # bad annotation
                'Subject_6/close_milk/3',       # bad annotation
                'Subject_1/give_card/3',        # bad annotation
                'Subject_1/receive_coin/3',     # bad annotation
                'Subject_1/receive_coin/1',     # bad annotation (bad at the end)
                'Subject_1/open_wallet/3',      # bad annotation (not so bad)
                'Subject_3/open_milk/3',        # bad annotation (blurry for small part)
                ]
    bad_index = _get_all_vid_idx(bad_seqs)

    all_bad_index = np.unique(np.concatenate((bad_index, outbound_index)))
    print(len(all_bad_index))

    print("WRITING BAD IMGS TO FILE")

    with open(parent_dir/'data'/'labels'/'fpha_img_invalid.txt', 'w') as f:
        for index in tqdm(all_bad_index):
            f.write("%s\n" % all_file_name[index])

if __name__ == '__main__':
    list_invalid_fpha()