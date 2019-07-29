import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import logging
import time

import dataset
import models
import models.znb_net as znbnet
from utils.directory import DATA_DIR
from utils.logger import get_logger, TBLogger
# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--print_freq', type=int, default=5)
parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--exp', type=str, default='')
parser.add_argument('--gpu_ids', type=str, default='[0]')
args = parser.parse_args()

gpu_ids = list(map(int, args.gpu_ids.strip('[]').split(',')))

# setup logging
logger = get_logger()

gpu_id_str = ''
for i in gpu_ids:
    gpu_id_str += str(i) + ' '

logger.info('USING GPU: ' + gpu_id_str)

if len(gpu_ids) == 1:
    cuda_gpu = 'cuda:%i' %gpu_ids[0]
else:
    cuda_gpu = 'cuda'

device = torch.device(cuda_gpu if torch.cuda.is_available() else 'cpu')

model = models.get_pose_net(logger=logger, gpu_ids=gpu_ids)

trainable_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)

print('no. of trainable parameters:', trainable_params)
input_tmp = torch.tensor(np.empty([1, 3, 256, 256]), dtype=torch.float32).to(device)
# input_tmp = torch.tensor(np.empty([1, 21, 32, 32]), dtype=torch.float32).to(device)
out = model.net(input_tmp)
for o in out:
    print('output shape:', o.shape)
