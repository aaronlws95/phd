import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_fpha_bbox():
    """ Prepare list for FPHA dataset """
    img_root    = Path(ROOT)/'datasets'/'First_Person_Action_Benchmark'
    skel_root   = 'Hand_pose_annotation_v1'
    video_root  = 'Video_files'
    modality    = 'color'
    img_size    = (1920, 1080)
    img_tmpl    = 'color_{:04d}.jpeg'
    pad         = 50
    
    # Accumulate list
    train_img_list  = []
    test_img_list   = []
    train_bbox_list = []
    test_bbox_list  = []
    with open(img_root/'data_split_action_recognition.txt') as f:
        cur_split = 'Training'
        lines = f.readlines()
        for l in tqdm(lines):
            words = l.split()
            if(words[0] == 'Training' or words[0] == 'Test'):
                cur_split = words[0]
            else:
                seq = words[0]
                seq_path = Path(seq)/modality
                full_path = img_root/video_root/seq_path
                len_frame_idx = len([x for x in full_path.glob('*') if x.is_file()])
                skeleton_path = img_root/skel_root/seq/'skeleton.txt'
                skel_list = np.loadtxt(skeleton_path)

                for i in range(len_frame_idx):
                    img_path = seq_path/img_tmpl.format(i)
                    skel_xyz = skel_list[:, 1:].reshape(skel_list.shape[0], 21, 3)[i]
                    skel_uvd = FPHA.xyz2uvd_color(skel_xyz)
                    bbox_gt = get_bbox_from_pose(skel_uvd, img_size=img_size, pad=pad)
                    if cur_split == 'Training':
                        train_bbox_list.append(bbox_gt)
                        train_img_list.append(img_path)
                    else:
                        test_bbox_list.append(bbox_gt)
                        test_img_list.append(img_path)

    # Save
    parent_dir = Path(__file__).absolute().parents[1]
    np.savetxt(parent_dir/'data'/'labels'/'fpha_bbox_pad{}_train.txt'.format(pad), train_bbox_list)
    np.savetxt(parent_dir/'data'/'labels'/'fpha_bbox_pad{}_test.txt'.format(pad), test_bbox_list)

if __name__ == '__main__':
    list_fpha_bbox()