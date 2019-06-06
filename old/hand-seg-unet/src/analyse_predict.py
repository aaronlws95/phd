import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import egohand_utils as ego
import hof_utils as hof
import time
from tqdm import tqdm
import keras.backend as K
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default=None, help='experiment directory name', required=True)
parser.add_argument('--load_epoch', type=int, default=0, help='0 to not load', required=False)
parser.add_argument('--data_split', type=str, default='train', choices=['train', 'val', 'test'], help='data to use: train, val or test', required=False)
parser.add_argument('--plot_threshold', type=bool, default=False, help='plot metric vs hard threshold', required=False)
parser.add_argument('--hard_threshold', type=float, default=None, help='set prediction > threshold = 1', required=False)
parser.add_argument('--dataset', type=str, default=None, help='name of dataset used', required=False)
parser.add_argument('--model_type', type=str, default=None, help='type of model used', required=False)
args = parser.parse_args()

experiment = args.experiment
data_split = args.data_split
load_epoch = args.load_epoch

if args.model_type:
    model_type = args.model_type
else:
    model_type = args.dataset

PATH_TO_WORKDIR = '/media/aaron/DATA/ubuntu/hand-seg/%s' %args.dataset
PATH_TO_MODELDIR = '/media/aaron/DATA/ubuntu/hand-seg/%s/' %model_type
PATH_TO_PREDICT = os.path.join(PATH_TO_MODELDIR, '%s/predict/predict_%s_%d_%s.h5' %(experiment, data_split, load_epoch, args.dataset))

if args.dataset == 'egohand':
    ori_dim = (720, 1280)
    resize_dim = (256, 256)
    jitter_dim = (224, 224)
if args.dataset == 'hand_over_face':
    ori_dim = (216, 384)
    resize_dim = (256, 256)
    jitter_dim = (224, 224)

def hard_threshold(img_batch, threshold):
    new_img_batch = np.copy(img_batch)
    if threshold is not None:
        # for img in new_img_batch:
        new_img_batch[new_img_batch > threshold] = 1
        new_img_batch[new_img_batch <= threshold] = 0
    return new_img_batch

def get_mask_predict(data_split):
    h5f = h5py.File(PATH_TO_PREDICT, 'r')
    predict = h5f['predict'][:]
    h5f.close()

    mask_path = os.path.join(PATH_TO_WORKDIR, 'mask_%s.h5' %data_split)
    h5f = h5py.File(mask_path, 'r')
    mask = h5f['mask'][:]
    h5f.close()

    predict = np.squeeze(predict, -1)
    return mask, predict

# Show images
def ego_show_predict_all(data_file, mask, predict):
    # MAX_FRAMES = 100
    max_num_frames = 5
    with open(data_file) as f:
        video_list = f.readlines()
        for j, video in enumerate(video_list):
            ref_video = video.strip()
            for k, frame in enumerate(ego.get_frames(ref_video)):
                if max_num_frames == k:
                    break

                i = j*100 + k

                img_gray = ego.get_img(ref_video, frame, resize_dim, is_gray=True)

                img_RGB = ego.get_img(ref_video, frame, resize_dim, is_gray=False)
                img_RGB = img_RGB[:, :, ::-1]
                img_RGB = cv2.normalize(img_RGB, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

                rgb_mask = cv2.cvtColor(mask[i], cv2.COLOR_GRAY2RGB)
                rgb_mask = cv2.normalize(rgb_mask, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                rgb_mask[:,:,0] = 0
                rgb_mask[:,:,2] = 0
                rgb_predict = cv2.cvtColor(predict[i], cv2.COLOR_GRAY2RGB)
                rgb_predict = cv2.normalize(rgb_predict, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                rgb_predict[:,:,0] = 0
                rgb_predict[:,:,2] = 0

                alpha = 0.5
                masked_over = (1-alpha)*img_RGB + alpha*rgb_mask
                predict_over = (1-alpha)*img_RGB + alpha*rgb_predict

                print('IoU: %f' %calc_iou(mask[i], predict[i]))
                print('Dice: %f' %calc_dice(mask[i], predict[i]))
                print('Precision: %f' %calc_precision(mask[i], predict[i]))
                print('Recall: %f' %calc_recall(mask[i], predict[i]))
                print('Binary Crossentropy: %f' %calc_bce(mask[i], predict[i]))
                print('----------------------')

                # fig, ax = plt.subplots()
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.imshow(predict_over)
                # plt.show()

                fig, ax = plt.subplots(2, 3)
                ax[0, 0].imshow(img_RGB)
                ax[0, 0].set_title('input RGB')
                ax[0, 1].imshow(img_gray, cmap='gray')
                ax[0, 1].set_title('input gray')
                ax[0, 2].imshow(predict[i], cmap='gray')
                ax[0, 2].set_title('predict')
                ax[1, 0].imshow(mask[i], cmap='gray')
                ax[1, 0].set_title('target')
                ax[1, 1].imshow(masked_over)
                ax[1, 1].set_title('input + target')
                ax[1, 2].imshow(predict_over)
                ax[1, 2].set_title('input + predict')
                plt.show()

def hof_show_predict_all(data_file, mask, predict):
    with open(data_file) as f:
        img_list = f.readlines()
        for i, img_id in enumerate(img_list):

            ref_img_id = int(img_id.strip())

            img_gray = hof.get_img(ref_img_id, resize_dim, is_gray=True)

            img_RGB = hof.get_img(ref_img_id, resize_dim, is_gray=False)
            img_RGB = img_RGB[:, :, ::-1]
            img_RGB = cv2.normalize(img_RGB, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            rgb_mask = cv2.cvtColor(mask[i], cv2.COLOR_GRAY2RGB)
            rgb_mask = cv2.normalize(rgb_mask, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            rgb_predict = cv2.cvtColor(predict[i], cv2.COLOR_GRAY2RGB)
            rgb_predict = cv2.normalize(rgb_predict, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            alpha = 0.4
            masked_over = (1-alpha)*img_RGB + alpha*rgb_mask
            predict_over = (1-alpha)*img_RGB + alpha*rgb_predict

            print('IoU: %f' %calc_iou(mask[i], predict[i]))
            print('Dice: %f' %calc_dice(mask[i], predict[i]))
            print('Precision: %f' %calc_precision(mask[i], predict[i]))
            print('Recall: %f' %calc_recall(mask[i], predict[i]))
            print('Binary Crossentropy: %f' %calc_bce(mask[i], predict[i]))
            print('----------------------')

            fig, ax = plt.subplots(2, 3)
            ax[0, 0].imshow(img_RGB)
            ax[0, 0].set_title('input RGB')
            ax[0, 1].imshow(img_gray, cmap='gray')
            ax[0, 1].set_title('input gray')
            ax[0, 2].imshow(predict[i], cmap='gray')
            ax[0, 2].set_title('predict')
            ax[1, 0].imshow(mask[i], cmap='gray')
            ax[1, 0].set_title('target')
            ax[1, 1].imshow(masked_over)
            ax[1, 1].set_title('input + target')
            ax[1, 2].imshow(predict_over)
            ax[1, 2].set_title('input + predict')
            plt.show()

def show_mask_predict(mask, predict):
    if(len(mask.shape) == 3):
        mask = np.squeeze(mask, -1)

    if(len(predict.shape) == 3):
            predict = np.squeeze(predict, -1)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mask, cmap='gray')
    ax[0].set_title('mask')
    ax[1].imshow(predict, cmap='gray')
    ax[1].set_title('predict')
    plt.show()

# Intersection over Union or Jaccard index
def calc_iou(mask, predict, smooth=1e-15):
    intersection = mask * predict
    union = mask + predict - intersection
    iou_score = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return iou_score

def calc_av_iou(mask, predict):
    num_data = mask.shape[0]
    mean_score = 0
    for i in range(num_data):
        mean_score += calc_iou(mask[i], predict[i])
    mean_score /= num_data
    return mean_score

# Dice similarity coefficient/F1 score
def calc_dice(mask, predict, smooth=1e-15):
    intersection = mask * predict
    dice_coeff = (2 * np.sum(intersection) + smooth) / (np.sum(mask) + np.sum(predict) + smooth)
    return dice_coeff

def calc_av_dice(mask, predict):
    num_data = mask.shape[0]
    mean_score = 0
    for i in range(num_data):
        mean_score += calc_dice(mask[i], predict[i])
    mean_score /= num_data
    return mean_score

# Precision
def calc_precision(mask, predict, smooth=1e-15):
    intersection = mask * predict
    precision = (np.sum(intersection) + smooth) / (np.sum(predict) + smooth)
    return precision

def calc_av_precision(mask, predict):
    num_data = mask.shape[0]
    mean_score = 0
    for i in range(num_data):
        mean_score += calc_precision(mask[i], predict[i])
    mean_score /= num_data
    return mean_score

# Recall
def calc_recall(mask, predict, smooth=1e-15):
    intersection = mask * predict
    recall = (np.sum(intersection) + smooth) / (np.sum(mask) + smooth)
    return recall

def calc_av_recall(mask, predict):
    num_data = mask.shape[0]
    mean_score = 0
    for i in range(num_data):
        mean_score += calc_recall(mask[i], predict[i])
    mean_score /= num_data
    return mean_score

# Binary Crossentropy
def calc_bce_keras(mask, predict):
    bce = K.binary_crossentropy(K.variable(mask), K.variable(predict))
    return (K.eval(K.mean(bce)))

def calc_bce(mask, predict):
    predict = np.clip(predict, 1e-7, 1-1e-7)
    # predict = np.maximum(np.minimum(predict, 1 - K.epsilon()), K.epsilon())
    bce = -(mask * np.log(predict) + (1 - mask) * np.log(1 - predict))
    return np.mean(bce)

def calc_av_bce(mask, predict):
    num_data = mask.shape[0]
    mean_score = 0
    for i in range(num_data):
        mean_score += calc_bce(mask[i], predict[i])
    mean_score /= num_data
    return mean_score

# Plot graphs
def plot_dice_vs_hard_thresh(mask, predict):
    threshold = np.arange(0.1, 1.0, 0.1)
    scores = []
    for t in tqdm(threshold):
        predict_t = hard_threshold(predict, t)
        scores.append(calc_av_dice(mask, predict_t))

    fig, ax = plt.subplots()
    ax.plot(threshold, scores, marker='o', color='b')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    plt.show()

def plot_iou_vs_hard_thresh(mask, predict):
    threshold = np.arange(0.1, 1.0, 0.1)
    scores = []
    for t in tqdm(threshold):
        predict_t = hard_threshold(predict, t)
        scores.append(calc_av_iou(mask, predict_t))

    fig, ax = plt.subplots()
    ax.plot(threshold, scores, marker='o', color='b')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Jaccard Index')
    plt.show()

def plot_bce_vs_hard_thresh(mask, predict):
    threshold = np.arange(0.1, 1.0, 0.1)
    scores = []
    for t in tqdm(threshold):
        predict_t = hard_threshold(predict, t)
        scores.append(calc_av_bce(mask, predict_t))

    fig, ax = plt.subplots()
    ax.plot(threshold, scores, marker='o', color='b')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Binarry Cross-entropy')
    plt.show()

def plot_precision_recall(mask, predict):
    threshold = np.arange(0.1, 1.0, 0.1)
    precision = []
    recall = []
    for t in tqdm(threshold):
        predict_t = hard_threshold(predict, t)
        precision.append(calc_av_precision(mask, predict_t))
        recall.append(calc_av_recall(mask, predict_t))

    fig, ax = plt.subplots()
    fig.suptitle('Varying Threshold')
    ax.plot(recall, precision, marker='o', color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.show()

print('%s' %data_split)
mask, predict = get_mask_predict(data_split)
if args.plot_threshold:
    plot_bce_vs_hard_thresh(mask, predict)
    plot_dice_vs_hard_thresh(mask, predict)
    plot_iou_vs_hard_thresh(mask, predict)
    plot_precision_recall(mask, predict)
predict = hard_threshold(predict, args.hard_threshold)
print('Average IoU', calc_av_iou(mask, predict))
print('Average Dice', calc_av_dice(mask, predict))
print('Average Precision', calc_av_precision(mask, predict))
print('Average Recall', calc_av_recall(mask, predict))
print('Average Binary Cross-entropy:', calc_av_bce(mask, predict))
data_file = os.path.join(PATH_TO_WORKDIR, 'data_%s.txt' %data_split)

if args.dataset == 'egohand':
    ego_show_predict_all(data_file, mask, predict)
if args.dataset == 'hand_over_face':
    hof_show_predict_all(data_file, mask, predict)
