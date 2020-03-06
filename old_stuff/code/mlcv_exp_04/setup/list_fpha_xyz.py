import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_fpha_xyz():
    img_root = Path(ROOT)/'datasets'/'First_Person_Action_Benchmark'
    skel_root = 'Hand_pose_annotation_v1'
    video_root = 'Video_files'
    modality = 'color'
    img_tmpl = 'color_{:04d}.jpeg'

    # Accumulate list
    train_list  = []
    test_list   = []
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
                    skel_xyz = skel_list[:, 1:][i]
                    if cur_split == 'Training':
                        train_list.append(skel_xyz)
                    else:
                        test_list.append(skel_xyz)

    # Save
    parent_dir = Path(__file__).absolute().parents[1]
    np.savetxt(parent_dir/'data'/'labels'/'fpha_xyz_train.txt', train_list)
    np.savetxt(parent_dir/'data'/'labels'/'fpha_xyz_test.txt', test_list)

if __name__ == '__main__':
    list_fpha_xyz()