import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.io

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT
from src.utils import *

def list_egohands_bbox():
    img_root = Path(ROOT)/'datasets'/'egohands'
    img_path = img_root/'_LABELLED_SAMPLES'
    metadata = scipy.io.loadmat(img_root/'metadata.mat')
    metadata_video = metadata['video'][0]
    num_videos = len(metadata_video)
    #mode = '' # all
    # mode = '_ego'
    mode = '_ego_all'
    if mode == '':
        hand_types = ['myleft', 'myright', 'yourleft', 'yourright']
    elif '_ego' in mode:
        hand_types = ['myleft', 'myright']

    train_img_list  = []
    train_bbox_list  = []
    val_img_list    = []
    val_bbox_list    = []
    pad = 10
    for vid_id in range(num_videos):
        # video.dtype: video_id, partner_vide_id, ego_viewer_id, partner_id, location_id, activity_id, labelled_frames
        cur_video = metadata_video[vid_id]
        video_path = img_path/str(cur_video[0][0])

        metadata_frame = cur_video['labelled_frames'][0]
        num_frames = len(metadata_frame)

        for frame_idx in range(num_frames):
            # frame.dtype: frame_num, myleft, myright, yourleft, yourright
            cur_frame = metadata_frame[frame_idx]
            img_file = video_path/'frame_{:04d}.jpg'.format(cur_frame['frame_num'][0][0])

            for hand_owner in hand_types:
                mask = cur_frame[hand_owner]
                if mask.size != 0:
                    bbox = get_bbox_from_mask_pts(mask, pad=pad)
                    # print(bbox)
                    img_file_save = str(img_file).split('/')
                    img_file_save = img_file_save[-2] + '/' + img_file_save[-1]
                    if mode == '_ego_all':
                        train_img_list.append(img_file_save)
                        train_bbox_list.append([float(x) for x in bbox])
                    else:
                        if frame_idx != 99:
                            train_img_list.append(img_file_save)
                            train_bbox_list.append([float(x) for x in bbox])
                        else:
                            val_img_list.append(img_file_save)
                            val_bbox_list.append([float(x) for x in bbox])

                    print(img_file)
                    fig, ax = plt.subplots()
                    img = cv2.imread(str(img_file))[:, :, ::-1]
                    ax.imshow(img)
                    draw_bbox(ax, bbox)
                    plt.show()

    # parent_dir = Path(__file__).absolute().parents[1]
    # np.savetxt(parent_dir/'data'/'labels'/'egohand_bbox_pad{}{}_train.txt'.format(pad, mode), train_bbox_list)
    # with open(parent_dir/'data'/'labels'/'egohand_img{}_train.txt'.format(mode), 'w') as f:
    #     for i in train_img_list:
    #         f.write("%s\n" %i)
    # np.savetxt(parent_dir/'data'/'labels'/'egohand_bbox_pad{}{}_val.txt'.format(pad, mode), val_bbox_list)
    # with open(parent_dir/'data'/'labels'/'egohand_img{}_val.txt'.format(mode), 'w') as f:
    #     for i in val_img_list:
    #         f.write("%s\n" %i)

if __name__ == '__main__':
    list_egohands_bbox()