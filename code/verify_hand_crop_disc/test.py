import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.models as models
import random
from tensorboardX import SummaryWriter

from dataset import *
from discriminator import *

exp = 'exp3'
epoch = 49

map_loc = 'cuda:0'
load_dir = Path(ROOT)/'verify_hand_crop_disc'/'data'/'{}'.format(exp)/'model_verify_hand_disc_{}.state'.format(epoch)
ckpt = torch.load(load_dir, map_location=map_loc)
model.load_state_dict(ckpt['model_state_dict'])

cfg = parse('/mnt/4TB/aaron/verify_hand_crop_disc/config_{}.cfg'.format(exp))

model_type = cfg['model']

if model_type == 'dcgan':
    model = DCGAN_Discriminator().cuda()

cfg['aug'] = False

dataset = FPHA_Hand_Crop_Dataset(cfg, 'test')

kwargs = {
    'batch_size'    : 1,
    'shuffle'       : cfg['shuffle'],
    'num_workers'   : int(cfg['num_workers']),
    'sampler'       : None,
    'pin_memory'    : True
}

dataloader = torch.utils.data.DataLoader(dataset, **kwargs)

total_no_hand = 0
total_hand = 0
TP = 0
FP = 0
FN = 0
TN = 0
thresh = 0.5
with torch.no_grad():
    model = model.eval()
    for step, (img_crop, is_hand) in enumerate(dataloader):
        print('{}/{}'.format(step, len(dataloader)))
        img_crop = img_crop.cuda()

        out = model(img_crop)
        out = out.squeeze()

        if out > thresh and is_hand == 1:
            TP += 1
        elif out > thresh and is_hand == 0:
            FP += 1
        elif out < thresh and is_hand == 1:
            FN += 1
        elif out < thresh and is_hand == 0:
            TN += 1

assert len(dataloader) == (TP+FP+FN+TN)
print('PRECISION:', TP/(TP + FP))
print('RECALL:', TP/(TP + FN))
print('ACCURACY', (TP + TN)/(TP+FP+TN+FN))

