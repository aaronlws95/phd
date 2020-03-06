import torch
import torch.nn as nn
from torchvision.transforms import Resize
import numpy as np
import sys
import os
import argparse
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.directory import DATA_DIR
from utils.logger import TBLogger
import model as m
import loss as l
import dataset as d
import hpo_net as hnet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--gpu_id', type=int, default=-1)
    args = parser.parse_args()

    # if mp.get_start_method(allow_none=True) != 'spawn':
        # mp.set_start_method('spawn')
        # print('SET MULTIPROCESSING TO SPAWN')
    if args.gpu_id == -1:
        print('USING CPU')
    else:
        print('USING GPU: %i' %args.gpu_id)
        torch.cuda.set_device(args.gpu_id)

    CKPT_DIR = os.path.join(DATA_DIR, args.exp, 'ckpt')
    load_dir = os.path.join(CKPT_DIR, 'model_%i.state' %args.epoch)
    ckpt = torch.load(load_dir)
    print('LOADED CHECKPOINT %i' %args.epoch)

    cfgfile = 'yolov2_hpo.cfg'
    net = hnet.HPO_Net(cfgfile)
    loss = l.HPOLoss()
    lr = 0.0001
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    scheduler = None
    model = m.Model(net, loss, optimizer, scheduler, gpu_id=args.gpu_id, ckpt=ckpt)

    for data_split in ['train', 'test']:
        print('PREDICTING IN DATA SPLIT: %s' %data_split)
        save_prefix = '%s_fpha' %data_split

        dataset = d.FPHA(save_prefix)

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=2)

        print('CREATED %s_LOADER' %data_split.upper())

        keypoints = []
        keypoints_best = []
        conf_vals = []
        count = 0
        model.net.eval()
        with torch.no_grad():
            for img, _, _, _ in tqdm(data_loader):
                count += 1
                if args.gpu_id != -1:
                    input_img = img.cuda()
                else:
                    input_img = img.cpu()
                _, pred_uvd, pred_conf = model.net(input_img)
                max_conf_idx = pred_conf.max(-1)[-1].cpu().numpy()
                keypoints_best.append(pred_uvd[0, max_conf_idx].cpu().numpy())
                conf_vals.append(pred_conf.cpu().numpy())
                if count <= 10:
                    keypoints.append(pred_uvd.cpu().numpy())

            WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'predict_%s_%s_uvd_845.txt' %(args.epoch, data_split))
            print('WRITING TO %s' %WRITE_DIR)

            with open(WRITE_DIR, "w") as f:
                    for pred in keypoints:
                        for hand in pred[0]:
                            for jnt in hand:
                                for dim_pt in jnt:
                                    f.write(str(dim_pt) + ' ')
                            f.write('\n')

            WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'predict_%s_%s_uvd.txt' %(args.epoch, data_split))
            print('WRITING TO %s' %WRITE_DIR)

            with open(WRITE_DIR, "w") as f:
                for pred in keypoints_best:
                    for jnt in np.squeeze(pred):
                        for dim_pt in jnt:
                            f.write(str(dim_pt) + ' ')
                    f.write('\n')

            WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'predict_%s_%s_conf.txt' %(args.epoch, data_split))
            print('WRITING TO %s' %WRITE_DIR)

            with open(WRITE_DIR, "w") as f:
                for conf in conf_vals:
                    for val in conf[0]:
                        f.write(str(val) + ' ')
                    f.write('\n')

if __name__ == '__main__':
    main()
