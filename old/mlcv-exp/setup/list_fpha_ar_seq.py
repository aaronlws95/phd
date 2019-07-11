import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT

def list_fpha_ar_seq():
    """ Prepare list for FPHA dataset """
    img_root = Path(ROOT)/'First_Person_Action_Benchmark'
    video_root = 'Video_files'
    modality = 'color'
    
    # Get noun labels
    action_to_noun  = {}
    noun_label      = []
    with open(img_root/'action_object_info.txt', 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                l = l.split(' ')
                noun_label.append(l[2])
                action_to_noun[int(l[0]) - 1] = l[2]
    noun_label = {k: v for v, k in enumerate(np.unique(noun_label))}

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
                action = int(words[1])
                seq_path = Path(seq)/modality
                full_path = (img_root/video_root/seq_path)
                len_frame_idx = len([x for x in full_path.glob('*') if x.is_file()])
                noun = noun_label[action_to_noun[action]]
                seq_info = (seq_path, len_frame_idx, action, noun)
                if cur_split == 'Training':
                    train_list.append(seq_info)
                else:
                    test_list.append(seq_info)

    # Save
    parent_dir = Path(__file__).absolute().parents[1]
    with open(parent_dir/'data'/'labels'/'fpha_ar_seq_train.txt', 'w') as f:
        f.write('\n'.join('{} {} {} {}'.format(x[0], x[1], x[2], x[3]) 
                          for x in train_list))
    with open(parent_dir/'data'/'labels'/'fpha_ar_seq_test.txt', 'w') as f:
        f.write('\n'.join('{} {} {} {}'.format(x[0], x[1], x[2], x[3]) 
                          for x in test_list))
        
if __name__ == '__main__':
    list_fpha_ar_seq()