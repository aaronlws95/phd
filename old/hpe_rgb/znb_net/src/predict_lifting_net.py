import torch
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
import lmdb
import pickle
import h5py
from skimage.transform import resize

import dataset
import models
from utils.directory import DATA_DIR
from utils.logger import get_logger
import dataset.FPHA_dataset as fpha
import utils.prepare_data as pd
import utils.error as error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--pred_smap', type=bool, default=False)
    parser.add_argument('--epoch_pose', type=int, default=0)
    parser.add_argument('--exp_pose', type=str, default='')
    args = parser.parse_args()

    logger = get_logger()

    # if mp.get_start_method(allow_none=True) != 'spawn':
    #     mp.set_start_method('spawn')
    #     logger.info('SET MULTIPROCESSING TO SPAWN')

    # load model
    CKPT_DIR = os.path.join(DATA_DIR, args.exp, 'ckpt')
    load_dir = os.path.join(CKPT_DIR, 'model_%i.state' %args.epoch)
    ckpt = torch.load(load_dir)
    model = models.get_lifting_net(logger=logger, gpu_id=args.gpu_id, ckpt=ckpt)
    logger.info('LOADED CHECKPOINT %i' %args.epoch)


    if args.gpu_id == -1:
        logger.info('USING CPU')
    else:
        logger.info('USING GPU: %i' %args.gpu_id)
        torch.cuda.set_device(args.gpu_id)

    for data_split in ['train', 'test']:

        model.net.eval()
        with torch.no_grad():
            save_prefix = '%s_fpha' %data_split

            if args.pred_smap:
                logger.info('PREDICTING FROM PREDICTED SCOREMAPS FROM POSE NET EPOCH: %i EXP: %s' %(args.epoch_pose, args.exp_pose))

                # create data loader
                fpha_dataset = fpha.FPHA_pred_smap_dataset(args.exp_pose, args.epoch_pose, data_split)
                data_loader = dataset.znb_pred_smap_fpha_dataloader(save_prefix,
                                                            dataset=fpha_dataset,
                                                            batch_size=1,
                                                            shuffle=False,
                                                            num_workers=2,
                                                            logger=logger)

                xyz_canon = []
                rot_mat = []
                for smap in tqdm(data_loader):
                    if args.gpu_id != -1:
                        scoremap_pred = smap.cuda()
                    else:
                        scoremap_pred = smap.cpu()

                    pred_xyz_canon, pred_rot_mat = model.net(scoremap_pred)

                    pred_xyz_canon = pred_xyz_canon.cpu().numpy()
                    pred_xyz_canon = np.reshape(pred_xyz_canon, -1)
                    xyz_canon.append(pred_xyz_canon)

                    pred_rot_mat = pred_rot_mat.cpu().numpy()
                    pred_rot_mat = np.reshape(pred_rot_mat, -1)
                    rot_mat.append(pred_rot_mat)

                WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'xyz_canon_%s_%s_smap_%s_%s.txt' %(args.epoch, data_split, args.epoch_pose, args.exp_pose))
                logger.info('WRITING TO %s' %WRITE_DIR)

                with open(WRITE_DIR, "w") as f:
                    for pred in xyz_canon:
                        for jnt  in np.squeeze(pred):
                            f.write(str(jnt) + ' ')
                        f.write('\n')

                WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'rot_mat_%s_%s_smap_%s_%s.txt' %(args.epoch, data_split, args.epoch_pose, args.exp_pose))
                logger.info('WRITING TO %s' %WRITE_DIR)

                with open(WRITE_DIR, "w") as f:
                    for pred in rot_mat:
                        for jnt  in np.squeeze(pred):
                            f.write(str(jnt) + ' ')
                        f.write('\n')

            else:
                logger.info('PREDICTING IN DATA SPLIT FROM GT: %s' %data_split)

                # create data loader
                fpha_dataset = fpha.FPHA_lifting_net_dataset(save_prefix)
                data_loader = dataset.znb_lift_net_fpha_dataloader(save_prefix,
                                                        dataset=fpha_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=2,
                                                        logger=logger)

                xyz_canon = []
                rot_mat = []
                for smap_gt, _, _ in tqdm(data_loader):
                    if args.gpu_id != -1:
                        scoremap_gt = smap_gt.cuda()
                    else:
                        scoremap_gt = smap_gt.cpu()

                    pred_xyz_canon, pred_rot_mat = model.net(scoremap_gt)

                    pred_xyz_canon = pred_xyz_canon.cpu().numpy()
                    pred_xyz_canon = np.reshape(pred_xyz_canon, -1)
                    xyz_canon.append(pred_xyz_canon)

                    pred_rot_mat = pred_rot_mat.cpu().numpy()
                    pred_rot_mat = np.reshape(pred_rot_mat, -1)
                    rot_mat.append(pred_rot_mat)

                WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'xyz_canon_%s_%s.txt' %(args.epoch, data_split))
                logger.info('WRITING TO %s' %WRITE_DIR)

                with open(WRITE_DIR, "w") as f:
                    for pred in xyz_canon:
                        for jnt  in np.squeeze(pred):
                            f.write(str(jnt) + ' ')
                        f.write('\n')

                WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'rot_mat_%s_%s.txt' %(args.epoch, data_split))
                logger.info('WRITING TO %s' %WRITE_DIR)

                with open(WRITE_DIR, "w") as f:
                    for pred in rot_mat:
                        for jnt  in np.squeeze(pred):
                            f.write(str(jnt) + ' ')
                        f.write('\n')

if __name__ == '__main__':
    main()
