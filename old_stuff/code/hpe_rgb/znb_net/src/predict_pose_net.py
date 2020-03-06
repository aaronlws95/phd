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
    parser.add_argument('--map_location', type=str, default=None)
    args = parser.parse_args()

    logger = get_logger()

    # if mp.get_start_method(allow_none=True) != 'spawn':
    #     mp.set_start_method('spawn')
    #     logger.info('SET MULTIPROCESSING TO SPAWN')

    if args.gpu_id == -1:
        logger.info('USING CPU')
    else:
        logger.info('USING GPU: %i' %args.gpu_id)
        torch.cuda.set_device(args.gpu_id)

    # load model
    CKPT_DIR = os.path.join(DATA_DIR, args.exp, 'ckpt')
    load_dir = os.path.join(CKPT_DIR, 'model_%i.state' %args.epoch)
    ckpt = torch.load(load_dir)
    model = models.get_pose_net(logger=logger, gpu_id=args.gpu_id, ckpt=ckpt)
    logger.info('LOADED CHECKPOINT %i' %args.epoch)

    for data_split in ['train', 'test']:
        logger.info('PREDICTING IN DATA SPLIT: %s' %data_split)
        save_prefix = '%s_fpha' %data_split
        keys_cache_file = os.path.join(DATA_DIR, save_prefix + '_keys_cache.p')
        keys = pickle.load(open(keys_cache_file, "rb"))

        dataroot_uvd_gt_scaled = os.path.join(DATA_DIR, save_prefix + '_uvd_gt_scaled.lmdb')
        uvd_gt_scaled_env = lmdb.open(dataroot_uvd_gt_scaled, readonly=True, lock=False, readahead=False, meminit=False)

        # create data loader
        fpha_dataset = fpha.FPHA_pose_net_dataset(save_prefix)
        data_loader = dataset.znb_pose_net_fpha_dataloader(save_prefix,
                                                dataset=fpha_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=2,
                                                logger=logger)

        model.net.eval()
        with torch.no_grad():
            # idx = 0
            # l2_loss = 0
            # keypoints_2d = []
            # count = 0
            saved_scoremap = []
            for img, _ in tqdm(data_loader):
                # count += 1

                if args.gpu_id != -1:
                    input_img = img.cuda()
                else:
                    input_img = img.cpu()
                pred_scoremap = model.net(input_img)
                pred_scoremap = pred_scoremap[-1][0].cpu().numpy()
                pred_scoremap = np.reshape(pred_scoremap, (pred_scoremap.shape[1], pred_scoremap.shape[2], pred_scoremap.shape[0]))
                # pred_scoremap = resize(pred_scoremap, (256, 256), order=3, preserve_range=True)

                # key = keys[idx]
                # uvd_gt_scaled = pd.read_lmdb(key, uvd_gt_scaled_env, np.float32, (21, 3))

                # scoremap_gt = pd.create_multiple_gaussian_map(uvd_gt_scaled, (256, 256))
                # l2_loss += error.scoremap_error(scoremap_gt, pred_scoremap)
                # kpt = pd.detect_keypoints_from_scoremap(pred_scoremap)
                # kpt = np.reshape(kpt, (-1))
                # keypoints_2d.append(kpt)

                # if count <= 10:
                saved_scoremap.append(pred_scoremap)

            # print('L2 scoremap error:', l2_loss/len(data_loader))

            SCOREMAP_DIR = os.path.join(DATA_DIR, args.exp, 'scoremap_%s_%s.h5' %(args.epoch, data_split))
            logger.info('SAVING SCOREMAP TO %s' %SCOREMAP_DIR)

            f = h5py.File(SCOREMAP_DIR, 'w')
            f.create_dataset('scoremap', data=saved_scoremap)
            f.close()

            # WRITE_DIR = os.path.join(DATA_DIR, args.exp, 'predict_%s_%s.txt' %(args.epoch, data_split))
            # logger.info('WRITING TO %s' %WRITE_DIR)

            # with open(WRITE_DIR, "w") as f:
            #     for pred in keypoints_2d:
            #         for jnt  in np.squeeze(pred):
            #             f.write(str(jnt) + ' ')
            #         f.write('\n')

if __name__ == '__main__':
    main()
