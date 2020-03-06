import scipy.io
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
root_path = '/media/aaron/DATA/ubuntu/egohands_data'
video_path = os.path.join(root_path, '_LABELLED_SAMPLES')
metadata_path = os.path.join(root_path, 'metadata.mat')
metadata = scipy.io.loadmat(metadata_path)
hand_owner_list = ['myleft', 'myright', 'yourleft', 'yourright']

def get_video_ids():
    video_id_list = []
    for video_meta in metadata['video'][0]:
        video_id_list.append(video_meta['video_id'][0])
    return video_id_list

def get_frames(video):
    frame_id_list = []
    for video_meta in metadata['video'][0]:
        if(video_meta['video_id'][0] == video):
            for frame_meta in video_meta['labelled_frames'][0]:
                frame_id_list.append(frame_meta['frame_num'][0][0])
    return frame_id_list

def get_pts(video, frame):

    frame_id_list = []
    for video_meta in metadata['video'][0]:
        if(video_meta['video_id'][0] == video):
            for frame_meta in video_meta['labelled_frames'][0]:
                if frame_meta['frame_num'][0][0] == frame:
                    for hand_owner in hand_owner_list:
                        frame_id_list.append(frame_meta[hand_owner])
                    break

    return frame_id_list

def get_all_pts(video, hand_owner):
    frame_id_list = []
    for video_meta in metadata['video'][0]:
        if(video_meta['video_id'][0] == video):
            for frame_meta in video_meta['labelled_frames'][0]:
                frame_id_list.append(frame_meta[hand_owner])
    return frame_id_list

def show_imgs():
    alpha = 50
    hand_types = {
                'myleft': (255, 0, 0, alpha),
                'myright': (0, 255, 0, alpha),
                'yourleft': (0, 0, 255, alpha),
                'yourright': (255, 0, 255, alpha)
                }

    for key in metadata:
        print(key)
    print(metadata['video'][0].shape)

    num_videos = len(metadata['video'][0])
    for vid_idx in range(num_videos):
        video_meta = metadata['video'][0][vid_idx]
        data_path = os.path.join(video_path, video_meta['video_id'][0])

        # print(video_meta.dtype)
        # print(video_meta['video_id'])
        # print(video_meta['partner_video_id'])
        # print(video_meta['ego_viewer_id'])
        # print(video_meta['partner_id'])
        # print(video_meta['location_id'])
        # print(video_meta['activity_id'])
        # print(video_meta['location_id'])
        # print('labelled_frames')
        # print(video_meta['labelled_frames'].shape)

        # num_frames = len(video_meta['labelled_frames'][0])
        num_frames = 2
        for frame_idx in range(num_frames):
            frame_meta = video_meta['labelled_frames'][0][frame_idx]

            # print(frame_meta.dtype)
            # print(frame_meta['frame_num'])

            frame_path = os.path.join(data_path, 'frame_%04d.jpg' %frame_meta['frame_num'][0])

            img = Image.open(frame_path)
            img_draw = ImageDraw.Draw(img, 'RGBA')
            for key, value in hand_types.items():
                pts = frame_meta[key]
                if len(pts) != 0:
                    print(len(pts))
                    tuple_pts = [tuple(pt) for pt in pts]
                    img_draw.polygon(tuple_pts, value)
            img.show()
            input('Press to continue...')

def create_target(video, frame, resize_dim=None):
    img = Image.new('1' ,size=(1280, 720))
    img_draw = ImageDraw.Draw(img)
    pts = get_pts(video, frame)
    for hand_mask in pts:
        if len(hand_mask) != 0:
            tuple_pts = [tuple(pt) for pt in hand_mask]
            img_draw.polygon(tuple_pts, fill=1)
    if resize_dim:
        img = img.resize(resize_dim)
    img = np.asarray(img, np.bool)

    return img

# show_imgs()
# print(get_frames('CARDS_LIVINGROOM_S_H'))
# img = create_target('CARDS_COURTYARD_B_T', 11, (512, 512))
# print(img.shape)
# print(img.dtype)
# print(np.where(img != 0))
# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
