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

exp = 'exp4'

logger = SummaryWriter(Path(ROOT)/'verify_hand_crop_disc'/'data'/'{}'.format(exp)/'logs')

cfg = parse('/mnt/4TB/aaron/verify_hand_crop_disc/config_{}.cfg'.format(exp))

train_dataset = FPHA_Hand_Crop_Dataset(cfg, 'train')
kwargs = {
    'batch_size'    : int(cfg['batch_size']),
    'shuffle'       : cfg['shuffle'],
    'num_workers'   : int(cfg['num_workers']),
    'sampler'       : None,
    'pin_memory'    : True
}

train_dataloader = torch.utils.data.DataLoader(train_dataset, **kwargs)

data_load = next(iter(train_dataloader))

idx = 0
img, is_hand = data_load
img = ImgToNumpy()(img)
img = img[idx].copy()
is_hand = is_hand[idx].numpy().copy()
fig, ax = plt.subplots()
ax.imshow(img)
plt.show()
print(is_hand)

img, is_hand = data_load
img = ImgToNumpy()(img)
fig, ax = plt.subplots(4, 4, figsize=(15, 15))
idx = 0
for i in range(4):
    for j in range(4):
        if idx >= img.shape[0]:
            break
        cur_img = img[idx].copy()
        ax[i, j].imshow(cur_img)
        idx += 1
plt.show()

model_type = cfg['model']

if model_type == 'dcgan':
    model = DCGAN_Discriminator().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
bceloss = torch.nn.BCELoss()

max_epoch = 500
model = model.train()
for epoch in range(max_epoch):
    for step, (img_crop, is_hand) in enumerate(train_dataloader):

        img_crop = img_crop.cuda()
        is_hand = is_hand.type(torch.FloatTensor).cuda()

        out = model(img_crop)
        out = out.squeeze()
        loss = bceloss(out, is_hand)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch: {}/{}'.format(epoch, max_epoch), 'step: {}/{}'.format(step, len(train_dataloader)), 'loss: {}'.format(loss))
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}
    logger.add_scalar('loss', loss.item(), epoch)
    torch.save(state, Path(ROOT)/'verify_hand_crop_disc'/'data'/'{}'.format(exp)/'model_verify_hand_disc_{}.state'.format(epoch))
