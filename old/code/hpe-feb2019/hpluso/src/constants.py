import os
import numpy as np

DATASET_DIR = '/home/aaron/First_Person_Action_Benchmark'
IMG_DIR = os.path.join(DATASET_DIR, 'Video_files')
SKEL_DIR = os.path.join(DATASET_DIR, 'Hand_pose_annotation_v1')
CKPT_DIR = '../data/ckpt'
LOG_DIR =  '../data/log'

FPHA_COLOR_WIDTH = 1920
FPHA_COLOR_HEIGHT = 1080
RSZ_WIDTH = 416
RSZ_HEIGHT = 416
CHANNEL = 3

CAM_EXTR = [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
[0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
[-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
[0, 0, 0, 1]]

CAM_INTR_COLOR = [[1395.749023, 0, 935.732544],
[0, 1395.749268, 540.681030],
[0, 0, 1]]

CAM_INTR_DEPTH = [[475.065948, 0,  315.944855],
[0, 475.065857, 245.287079],
[0, 0, 1]]


REORDER = [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]

W = 13
H = 13
D = 5
C_u = RSZ_WIDTH//W
C_v = RSZ_HEIGHT//H
C_z = 120 #mm
a = 2 #sharpness
d_th = 75 #distance threshold

EPOCHS = 200
# BATCH_SIZE=16
BATCH_SIZE=1
