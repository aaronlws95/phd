import cv2
import datetime
import os
import tensorflow as tf
from keras.models import load_model
import egohand_utils as ego
import hof_utils as hof
import matplotlib.pyplot as plt
import time
import argparse
from custom_loss import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default=None, help='experiment directory name', required=True)
parser.add_argument('--load_epoch', type=int, default=0, help='0 to not load', required=False)
parser.add_argument('--hard_threshold', type=int, default=None, help='set mask > threshold = 1', required=False)
parser.add_argument('--dataset', type=str, default=None, help='name of dataset used', required=False)
args = parser.parse_args()

PATH_TO_WORKDIR = '/media/aaron/DATA/ubuntu/hand-seg/%s' %args.dataset

resize_dim = (256, 256)
alpha = 0.5 # 1 = 100% mask

load_model_no = args.load_epoch
experiment = args.experiment
PATH_TO_MODEL = os.path.join(PATH_TO_WORKDIR, '%s/models/model-%d.hdf5' %(experiment, load_model_no))
load_model = load_model(PATH_TO_MODEL, custom_objects=custom_obj)

# source = 0 # webcam
# source = '../../chess1.mp4'
source = '../../jordan.mp4'

cap = cv2.VideoCapture(source)
cap_dim = (int(cap.get(3)), int(cap.get(4)))
cv2.namedWindow('Real Time Processing', cv2.WINDOW_NORMAL)
start = time.time()
num_frames = 0
while True:
    num_frames += 1
    ret, frame = cap.read()

    frame = cv2.resize(frame, resize_dim)

    # input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) #gray
    input_img = frame.astype(np.float32) #color
    input_img = input_img[:, :, ::-1] #color
    input_img = (input_img - np.mean(input_img))/ np.std(input_img)
    input_img = np.expand_dims(input_img, axis=0)
    # input_img = np.expand_dims(input_img, axis=-1) #gray
    mask = load_model.predict(input_img)
    mask = np.squeeze(mask, axis=(0,-1))
    if args.hard_threshold:
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    frame = cv2.normalize(frame, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    mask = cv2.normalize(mask, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    mask[:, :, 0] = 0
    mask[:, :, 2] = 0

    out_img = (1-alpha)*frame + alpha*mask
    out_img = cv2.resize(out_img, (256, 256))

    cv2.imshow('Real Time Processing', out_img)

    out_img = cv2.normalize(out_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC3)
    path = os.path.join(PATH_TO_WORKDIR, 'save/%d.jpg' %num_frames)
    cv2.imwrite(path, out_img)

    end = time.time()
    seconds = end - start
    fps = num_frames / seconds
    print("Elapsed time: ", seconds, "FPS: ", fps)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
