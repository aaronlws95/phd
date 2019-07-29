import random
import os
import sys
import shutil
import pandas as pd 
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *
 
def create_train_img_dir():
    modality        = 'rgb'
    data_split      = 'train'
    csv_file        = 'EPIC_train_action_labels.csv'
    epic_img_root   = Path(ROOT)/'EPIC_KITCHENS_2018'/'frames_rgb_flow'
    save_img_root   = Path(ROOT)/'EPIC_KITCHENS_2018'/'EK_frames'
    if not save_img_root.is_dir():
        save_img_root.mkdir()
        
    epic_label  = Path(ROOT)/'EPIC_KITCHENS_2018'/'annotations'/csv_file
    epic_label  = pd.read_csv(epic_label).to_numpy()
    
    for label_no in tqdm(range(len(epic_label))):
        cur_vid         = epic_label[label_no]
        uid             = cur_vid[0]
        subject_id      = cur_vid[1]
        seq             = cur_vid[2]
        start_frame     = cur_vid[6]
        end_frame       = cur_vid[7]
        verb            = cur_vid[8]
        noun            = cur_vid[10].split(':')[0]

        seq_path        = epic_img_root/modality/data_split/subject_id/seq
        all_frames      = [x for x in os.listdir(seq_path) if os.path.join(seq_path, x)]
        all_frames_idx  = [(x.split('_')[-1][:-3]) for x in all_frames]
        all_frames_idx  = np.argsort(all_frames_idx).astype('uint32')
        all_frames      = np.asarray(all_frames)[all_frames_idx]    
        image_idx       = all_frames[start_frame:end_frame]

        save_seq_dir    = save_img_root/subject_id/(seq + '_{}'.format(uid) + '_' + verb + '_' + noun)

        if not save_seq_dir.is_dir():
            save_seq_dir.mkdir(parents=True, exist_ok=True)

        # create new files
        for i, frame in enumerate(image_idx):
            cur_frame_path = epic_img_root/modality/data_split/subject_id/seq/frame
            save_frame_path = save_seq_dir/'img_{:05d}.jpg'.format(i)
            if save_frame_path.is_file():
                # print('Exists: ', save_frame_path)
                continue
            print('Copying: ', save_frame_path)
            shutil.copyfile(cur_frame_path, save_frame_path)

def create_lin_id(unique_ids):
    """ Arrange unique ids in linear range with no gaps """
    lin_id = dict()
    for i, id in enumerate(np.sort(unique_ids)):
        lin_id[id] = i
    return lin_id

def list_ek_ar_seq():
    modality        = 'rgb'
    data_split      = 'train'
    csv_file        = 'EPIC_train_action_labels.csv'
        
    epic_label      = Path(ROOT)/'EPIC_KITCHENS_2018'/'annotations'/csv_file
    epic_label      = pd.read_csv(epic_label)
    verb_lin_dict   = create_lin_id(np.sort(epic_label.verb_class.unique()))
    noun_lin_dict   = create_lin_id(np.sort(epic_label.noun_class.unique()))
    epic_label      = epic_label.to_numpy()
    
    seq_train_info_list     = []
    seq_val_info_list       = []
    seq_train_all_info_list = []
    video_indices = np.arange(len(epic_label))
    np.random.shuffle(video_indices)
    val_length      = 1000
    val_indices     = video_indices[:val_length]
    train_indices   = video_indices[val_length:]

    for split, indices in [('train', train_indices), ('val', val_indices)]:
        for label_no in tqdm(indices):
            cur_vid         = epic_label[label_no]
            uid             = cur_vid[0]
            subject_id      = cur_vid[1]
            seq             = cur_vid[2]
            start_frame     = cur_vid[6]
            end_frame       = cur_vid[7]
            verb            = cur_vid[8]
            verb_class      = cur_vid[9]
            noun            = cur_vid[10].split(':')[0]
            noun_class      = cur_vid[11]
            
            save_seq_dir    = Path(subject_id)/(seq + '_{}'.format(uid) + '_' + verb + '_' + noun)
            
            len_seq         = end_frame - start_frame 
            if split == 'train':
                seq_train_info_list.append((save_seq_dir, 
                                            len_seq, 
                                            verb_lin_dict[verb_class], 
                                            noun_lin_dict[noun_class]))
            elif split == 'val':
                seq_val_info_list.append((save_seq_dir, 
                                          len_seq, 
                                          verb_lin_dict[verb_class], 
                                          noun_lin_dict[noun_class]))

    parent_dir = Path(__file__).absolute().parents[1]
    with open(parent_dir/'data'/'labels'/'ek_ar_seq_train.txt', 'w') as f:
        f.write('\n'.join('{} {} {} {}'.format(x[0], x[1], x[2], x[3]) for x in seq_train_info_list))
    with open(parent_dir/'data'/'labels'/'ek_ar_seq_test.txt', 'w') as f:
        f.write('\n'.join('{} {} {} {}'.format(x[0], x[1], x[2], x[3]) for x in seq_val_info_list))

if __name__ == '__main__':
    list_ek_ar_seq()