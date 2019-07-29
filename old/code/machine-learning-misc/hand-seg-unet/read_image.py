import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import image_utils as imgut
import read_meta as rm

root_path = '/media/aaron/DATA/ubuntu/egohands_data'
video_path = os.path.join(root_path, '_LABELLED_SAMPLES')
ckpt_path = os.path.join(root_path, 'checkpoint')
data_path = os.path.join(root_path, 'data')
eval_path = os.path.join(root_path, 'eval')

exp_no = '1'
datatype = 'test'

def get_masked_img(exp_no, datatype, video, frame_num, img_type, alpha):
    img_path = os.path.join(eval_path, exp_no)
    img_path = os.path.join(img_path, datatype)
    img_path = os.path.join(img_path, video)

    img = os.path.join(img_path, 'frame_%04d_input.png' %frame_num)
    img = cv2.imread(img)

    mask = os.path.join(img_path, 'frame_%04d_%s.png' %(frame_num, img_type))
    mask = cv2.imread(mask)
    mask = cv2.normalize(mask, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    mask = mask.astype(np.uint8)

    out_img = np.zeros(img.shape, dtype=img.dtype)
    out_img[:,:,:] = (alpha * img[:,:,:]) + ((1-alpha) * mask[:,:,:])
    return out_img

def show_all_img(exp_no, datatype):
    img_path = os.path.join(eval_path, exp_no)
    img_path = os.path.join(img_path, datatype)
    chosen_data_path = os.path.join(data_path, 'data_%s.txt' %datatype)
    video_list = []
    alpha = 0.4
    with open(chosen_data_path) as f:
        video_list = f.readlines()

    for video in video_list:
        video = video.strip()
        frames = rm.get_frames(video)
        video_path = os.path.join(img_path, video)
        for i, frame in enumerate(frames):
            if i>2:
                break
            predict = get_masked_img(exp_no, datatype, video, frame, 'predict', alpha)
            target = get_masked_img(exp_no, datatype, video, frame, 'target', alpha)
            imgut.show_predict_and_target(predict, target)

show_all_img(exp_no, datatype)
