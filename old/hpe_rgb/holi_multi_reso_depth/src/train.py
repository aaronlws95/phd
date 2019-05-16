import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import logging

from dataset import multireso_fpha_dataloader
from models import get_multireso_model
from utils.directory import DATA_DIR
from utils.logger import get_logger, TBLogger
# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--print_freq', type=int, default=5)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--exp', type=str, default='')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# setup logging
tb_logger = TBLogger(os.path.join(DATA_DIR, args.exp, 'log'))
logger = get_logger()

model = get_multireso_model(device)

# create data loader
train_h5py = os.path.join(DATA_DIR, 'train_fpha.h5')
train_loader = multireso_fpha_dataloader(train_h5py,
                                         batch_size=args.batch_size)
test_h5py = os.path.join(DATA_DIR, 'test_fpha.h5')
test_loader = multireso_fpha_dataloader(test_h5py,
                                        batch_size=args.batch_size)

CKPT_DIR = os.path.join(DATA_DIR, args.exp, 'ckpt')
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# load data
init_epoch = 0
if args.epoch != 0:
    load_dir = os.path.join(CKPT_DIR, 'model_%i.state' %args.epoch)
    ckpt = torch.load(load_dir)
    model.load_ckpt(ckpt)
    init_epoch = ckpt['epoch']
    logger.info('LOADED CHECKPOINT %i' %init_epoch)

total_step = len(train_loader)
cur_val_loss = 0
for epoch in range(init_epoch, args.num_epochs):
    for cur_step, (imgs, uvd_norm_gt) in enumerate(train_loader):
        # train
        img0 = imgs[0].to(device)
        img1 = imgs[1].to(device)
        img2 = imgs[2].to(device)
        uvd_norm_gt = uvd_norm_gt.to(device)

        out = model.net(img0,img1,img2)
        loss = model.loss(out, uvd_norm_gt)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        # log
        if (cur_step+1)%args.print_freq == 0:
            logger.info('[epoch %i, step %i/%i, loss: %f, val_loss: %f]' %(epoch+1, cur_step+1, total_step, loss.item(), cur_val_loss))

    # report validation loss
    model.net.eval()
    with torch.no_grad():
        val_loss = 0
        for imgs, uvd_norm_gt in test_loader:
            img0 = imgs[0].to(device)
            img1 = imgs[1].to(device)
            img2 = imgs[2].to(device)
            uvd_norm_gt = uvd_norm_gt.to(device)
            out = model.net(img0,img1,img2)
            val_loss += model.loss(out, uvd_norm_gt).item()
        val_loss /= len(test_loader)
        cur_val_loss = val_loss
    model.net.train()

    # tensorboard logging
    tb_log_info = {'loss': loss.item(), 'val_loss' : val_loss}
    for tag, value in tb_log_info.items():
        tb_logger.scalar_summary(tag, value, epoch+1)

    # save checkpoint
    if (epoch+1)%args.save_freq == 0:
            state = {'epoch': epoch, 'model_state_dict': model.net.state_dict(), 'optimizer': [], 'optimizer_state_dict': model.optimizer.state_dict(), 'loss': loss,}
            torch.save(state, os.path.join(CKPT_DIR, 'model_%i.state' %(epoch+1)))


