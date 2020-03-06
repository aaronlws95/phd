import cv2
import datetime
import os
import tensorflow as tf
from keras.models import load_model
import image_utils as imgut
import numpy as np

root_path = '/media/aaron/DATA/ubuntu/egohands_data'
video_path = os.path.join(root_path, '_LABELLED_SAMPLES')
ckpt_path = os.path.join(root_path, 'checkpoint')
data_path = os.path.join(root_path, 'data')
eval_path = os.path.join(root_path, 'eval')

load_model_no = 100
exp_no = '1'
dim = (256, 256)
alpha = 0.3 # 0 = 100% mask

cap = cv2.VideoCapture(0)
im_dim = (int(cap.get(3)), int(cap.get(4)))
cv2.namedWindow('Real Time Processing', cv2.WINDOW_NORMAL)

ckpt_path = os.path.join(ckpt_path, exp_no)
load_model = load_model(os.path.join(ckpt_path, 'model-%02d.hdf5' %load_model_no))

while True:
    ret, frame = cap.read()
    # try:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # except:
    #     print("Error converting to RGB")

    input_img = imgut.preprocess(frame, dim, 3)
    input_img = imgut.restore(input_img, dim, 3)
    input_img = np.expand_dims(input_img, axis=0)
    mask = load_model.predict(input_img)

    frame = cv2.resize(frame, dim)

    mask = np.squeeze(mask)
    RGB_mask = np.zeros_like(frame, dtype=np.float32)
    RGB_mask[:,:,0] = mask
    RGB_mask[:,:,1] = mask
    RGB_mask[:,:,2] = mask
    mask = RGB_mask

    mask = mask*255
    mask = mask.astype(np.uint8)

    out_img = np.zeros(frame.shape, dtype=frame.dtype)
    out_img[:,:,:] = (alpha * frame[:,:,:]) + ((1-alpha) * mask[:,:,:])

    out_img = cv2.resize(out_img, im_dim)

    cv2.imshow('Real Time Processing', out_img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
