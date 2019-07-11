import os
import pandas                                   as pd 
import numpy                                    as np
from pathlib                import Path
from tqdm                   import tqdm
from src.utils              import DATA_DIR

ROOT = Path(DATA_DIR)/'EPIC_KITCHENS_2018'

# ========================================================
# UTILITIES
# ========================================================

def get_img_path(subject_id,
                 seq_id,
                 frame_id,
                 dom='action' ,
                 modality='rgb', 
                 data_split='train'):
    if dom == 'action':
        epic_img_root = ROOT/'frames_rgb_flow'

    frame       = 'frame_{:010d}'.format(frame_id)
    subject     = 'P{:02d}'.format(subject_id)
    seq         = subject + '_{:02d}'.format(seq_id)
    img_path    = epic_img_root/modality/data_split/subject/seq/(frame + '.jpg')
    return img_path

def get_video_frames(idx, dom='action',
                     modality='rgb',
                     data_split='train'):
    if dom == 'action':
        csv_file        = 'EPIC_{}_action_labels.csv'.format(data_split)
        epic_img_root   = ROOT/'frames_rgb_flow'

    epic_label  = os.path.join(ROOT, 'annotations', csv_file)
    epic_label  = pd.read_csv(epic_label)
    cur_vid     = epic_label.loc[epic_label['uid'] == idx].to_numpy()[0]
    subject_id  = cur_vid[1]
    seq         = cur_vid[2]
    action      = cur_vid[3]
    start_frame = cur_vid[6]
    end_frame   = cur_vid[7]
    
    seq_path        = epic_img_root/modality/data_split/subject_id/seq
    all_frames      = [x for x in os.listdir(seq_path) if os.path.join(seq_path, x)]
    all_frames_idx  = [(x.split('_')[-1][:-3]) for x in all_frames]
    all_frames_idx  = np.argsort(all_frames_idx).astype('uint32')
    all_frames      = np.asarray(all_frames)[all_frames_idx]    
    image_id        = all_frames[start_frame:end_frame]
    
    img_path = []
    for idx in tqdm(image_id):
        img_path.append(epic_img_root/modality/data_split/subject_id/seq/idx)
    return img_path

def get_class_name(id, type):
    csv_file        = 'EPIC_{}_classes.csv'.format(type)
    csv_file        = os.path.join(ROOT, 'annotations', csv_file)
    classes         = pd.read_csv(csv_file)
    name            = classes.loc[classes['{}_id'.format(type)] == id].to_numpy()[0, 1]
    return name

def create_lin_id(unique_ids):
    """ Arrange unique ids in linear range with no gaps """
    lin_id = dict()
    for i, id in enumerate(np.sort(unique_ids)):
        lin_id[id] = i
    return lin_id

def rev_lin_id(unique_ids):
    """ Get dict to find actual id """
    lin_id = dict()
    for i, id in enumerate(np.sort(unique_ids)):
        lin_id[i] = id
    return lin_id