import torch
import argparse
import os
from tqdm import tqdm
import numpy as np

import dataset
import models
from utils.directory import DATA_DIR
from utils.logger import get_logger

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--exp', type=str, default='')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger = get_logger()

model = models.get_multireso_batchnorm_model(device, logger=logger)

# load model
CKPT_DIR = os.path.join(DATA_DIR, args.exp, 'ckpt')
load_dir = os.path.join(CKPT_DIR, 'model_%i.state' %args.epoch)
ckpt = torch.load(load_dir)
model.load_ckpt(ckpt)
logger.info('LOADED CHECKPOINT %i' %args.epoch)

for data_split in ['train', 'test']:
    logger.info('PREDICTING IN DATA SPLIT: %s' %data_split)

    # create data loader
    file_h5py = os.path.join(DATA_DIR, '%s_fpha_RGB.h5' %data_split)
    data_loader = dataset.multireso_fpha_batchnorm_dataloader(file_h5py,
                                            batch_size=1,
                                            shuffle=False,
                                            logger=logger,
                                            mode='test')

    model.net.eval()
    with torch.no_grad():
        out = []
        for imgs, uvd_norm_gt in tqdm(data_loader):
            img0 = imgs[0].to(device)
            img1 = imgs[1].to(device)
            img2 = imgs[2].to(device)
            uvd_norm_gt = uvd_norm_gt.to(device)
            pred_normuvd = model.net(img0,img1,img2)
            out.append(pred_normuvd.cpu().numpy())

    WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'predict_%s_%s.txt' %(args.epoch, data_split))
    logger.info('WRITING TO %s' %WRITE_DIR)

    with open(WRITE_DIR, "w") as f:
        for pred in out:
            for jnt in np.squeeze(pred):
                f.write(str(jnt) + ' ')
            f.write('\n')

