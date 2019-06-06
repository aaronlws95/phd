import os
import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw
from tqdm import tqdm
import h5py

PATH_TO_WORKDIR = '/media/aaron/DATA/ubuntu/hand-seg/egohand'
PATH_TO_DATA = '/media/aaron/DATA/ubuntu/egohands_data'
PATH_TO_VIDEOS= os.path.join(PATH_TO_DATA, '_LABELLED_SAMPLES')
metadata = scipy.io.loadmat(os.path.join(PATH_TO_DATA, 'metadata.mat'))
hand_owners = ['myleft', 'myright', 'yourleft', 'yourright']

# read from metadata
def get_video_ids():
    video_ids = []
    for video in metadata['video'][0]:
        video_ids.append(video['video_id'][0])
    return video_ids

def get_frames(ref_video):
    frame_ids = []
    for video in metadata['video'][0]:
        if(video['video_id'][0] == ref_video):
            for frame in video['labelled_frames'][0]:
                frame_ids.append(frame['frame_num'][0][0])
    return frame_ids

def get_mask_pts(ref_video, ref_frame):
    mask_pts = []
    for video in metadata['video'][0]:
        if(video['video_id'][0] == ref_video):
            for frame in video['labelled_frames'][0]:
                if frame['frame_num'][0][0] == ref_frame:
                    for hand_owner in hand_owners:
                        mask_pts.append(frame[hand_owner])
                    break
    return mask_pts

# show images
def show_all_masked_img():
    alpha = 0.4
    hand_types = {  'myleft'   : (1, 0, 0, alpha),
                    'myright'  : (0, 1, 0, alpha),
                    'yourleft' : (0, 0, 1, alpha),
                    'yourright': (1, 0, 1, alpha) }

    # for key in metadata:
        # print(key)
    # print(metadata['video'][0].shape)

    num_videos = len(metadata['video'][0])
    for vid_id in range(num_videos):
        # video.dtype: video_id, partner_vide_id, ego_viewer_id, partner_id, location_id, activity_id, labelled_frames
        video = metadata['video'][0][vid_id]
        video_path = os.path.join(PATH_TO_VIDEOS, video['video_id'][0])

        # num_frames = len(video['labelled_frames'][0]) = 100
        num_frames = 2 # display 2 frames per video
        for frame_idx in range(num_frames):
            # frame.dtype: frame_num, myleft, myright, yourleft, yourright
            frame = video['labelled_frames'][0][frame_idx]

            # plot
            fig, ax = plt.subplots()
            img = cv2.imread(os.path.join(video_path, 'frame_%04d.jpg' %frame['frame_num'][0]))
            hand_masks = []
            for hand_owner, mask_color in hand_types.items():
                mask = frame[hand_owner]
                if len(mask) != 0:
                    mask = Polygon(mask, True, color=mask_color)
                    ax.add_patch(mask)
            plt.imshow(img[:, :, ::-1]) # BGR --> RGB
            plt.show()

def show_masked_img(ref_video, ref_frame, resize_dim=None, is_gray=False):

    # mask
    mask = get_mask(ref_video, ref_frame, resize_dim)

    # img
    img = get_img(ref_video, ref_frame, resize_dim, is_gray)

    cmap_img = 'gray'
    if not is_gray:
        img = img[: , :, ::-1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB).astype(np.float32)
        cmap_img = None

    img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    # add mask and frame
    alpha = 0.5
    out_img = (1-alpha)*img + alpha*mask

    # plot
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap=cmap_img)
    ax[0].set_title('img')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('mask')
    ax[2].imshow(out_img, cmap=cmap_img)
    ax[2].set_title('img+mask')
    plt.show()

def show_mask_and_img(img, mask):
    alpha = 0.2

    # if RGB
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    mask = cv2.normalize(mask, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    out_img = (1-alpha)*img + alpha*mask

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('img')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('mask')
    ax[2].imshow(out_img, cmap='gray')
    ax[2].set_title('img+mask')
    plt.show()

# get images
def get_mask(ref_video, ref_frame, resize_dim=None):
    img = Image.new('1' ,size=(1280, 720)) #1 bit
    img_draw = ImageDraw.Draw(img)
    pts = get_mask_pts(ref_video, ref_frame)
    for hand_mask in pts:
        if len(hand_mask) != 0:
            tuple_pts = [tuple(pt) for pt in hand_mask]
            img_draw.polygon(tuple_pts, fill=1)
    if resize_dim:
        img = img.resize(resize_dim)

    img = np.asarray(img, np.float32)
    return img

def get_img(ref_video, ref_frame, resize_dim=None, is_gray=False):
    # img
    img = cv2.imread(os.path.join(PATH_TO_VIDEOS, '%s/frame_%04d.jpg' %(ref_video, ref_frame)))
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize_dim:
        img = cv2.resize(img, resize_dim)
    img = np.asarray(img, np.float32)
    return img

def save_mask_data_split(data_split):
    def get_mask_data_split(data_file, resize_dim=(256, 256)):
        mask = []
        with open(data_file) as f:
            video_list = f.readlines()
            for video in tqdm(video_list):
                ref_video = video.strip()
                for frame in get_frames(ref_video):
                    mask.append(get_mask(ref_video, frame, resize_dim))
        mask = np.asarray(mask)
        return mask

    data_file = os.path.join(PATH_TO_WORKDIR, 'data_%s.txt' %data_split)
    save_path = os.path.join(PATH_TO_WORKDIR, 'mask_%s.h5' %data_split)
    print('saving mask to %s' %save_path)
    h5f = h5py.File(save_path, 'w')
    mask = get_mask_data_split(data_file)
    h5f.create_dataset('mask', data=mask)
    h5f.close()

def get_bbox_pts(points):
    max_pts = np.amax(points, 0)
    min_pts = np.amin(points, 0)
    max_x = max_pts[0]
    max_y = max_pts[1]
    min_x = min_pts[0]
    min_y = min_pts[1]
    return max_y, min_y, max_x, min_x

#TEST
# ref_video = 'CARDS_LIVINGROOM_S_H'
# ref_frame = 3
# print('get_video_ids:', get_video_ids())
# print('get_frames:', get_frames(ref_video))
# show_mask_and_img(get_img(ref_video, ref_frame), get_mask(ref_video, ref_frame))
# show_masked_img(ref_video, ref_frame, (512, 512))
# show_masked_img(ref_video, ref_frame, (512, 512), True)
# show_all_masked_img()
