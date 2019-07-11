import random
import os
import sys
import numpy                                    as np
from pathlib                import Path
from tqdm                   import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src.utils import FPHA, DATA_DIR

def get_train_val_list():
    modality        = 'color'
    img_root        = Path(DATA_DIR)/'First_Person_Action_Benchmark'/'Video_files_rsz'
    
    seq_train_info_list = []
    seq_val_info_list   = []
    with open(os.path.join(DATA_DIR, 'First_Person_Action_Benchmark',
                           'data_split_action_recognition.txt')) as f:
        cur_split = 'Training'
        lines = f.readlines()
        for l in tqdm(lines):
            words = l.split()
            if(words[0] == 'Training' or words[0] == 'Test'):
                cur_split = words[0]
            else:
                path = l.split()[0]
                label = l.split()[1]
                full_path = os.path.join(img_root, path, modality)
                len_frame_idx = len([x for x in os.listdir(full_path)
                                    if os.path.join(full_path, x)])
                seq_info = (full_path, len_frame_idx, label)
                if cur_split == 'Training':
                    seq_train_info_list.append(seq_info)
                else:
                    seq_val_info_list.append(seq_info)

    with open(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'fpha_{}_action_train_rsz_list.txt'.format(modality), 'w') as f:
        f.write('\n'.join('{} {} {}'.format(x[0], x[1], x[2]) for x in seq_train_info_list))
    with open(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'fpha_{}_action_test_rsz_list.txt'.format(modality), 'w') as f:
        f.write('\n'.join('{} {} {}'.format(x[0], x[1], x[2]) for x in seq_val_info_list))

if __name__ == '__main__':
    get_train_val_list()