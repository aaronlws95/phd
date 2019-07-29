import torch.utils.data
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.prepare_data as pd
from .FPHA_dataset import FPHA_dataset
from utils.logger import get_logger

def multireso_fpha_dataloader(h5py_file, batch_size=32, shuffle=True, num_workers=12, logger=None):
    if logger:
      logger.info('CREATING DATA LOADER FOR: %s' %(h5py_file))
      logger.info('BATCH_SIZE: %i  SHUFFLE: %i' %(batch_size, shuffle))

    img0, img1, img2, uvd_norm_gt, _, _, _, _, = pd.read_data_h5py(h5py_file)
    img0 = pd.move_channel_dim_3_to_1(img0).astype('float32')
    img1 = pd.move_channel_dim_3_to_1(img1).astype('float32')
    img2 = pd.move_channel_dim_3_to_1(img2).astype('float32')
    uvd_norm_gt = np.reshape(uvd_norm_gt, (-1, 63))
    fpha_dataset = FPHA_dataset(img0, img1, img2, uvd_norm_gt)

    if logger:
      logger.info('DATA SIZE: %i' %(img0.shape[0]))

    return torch.utils.data.DataLoader(dataset=fpha_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)

def multireso_fpha_batchnorm_dataloader(h5py_file, batch_size=32, shuffle=True, num_workers=12, logger=None, mode='train'):
    if logger:
      logger.info('CREATING DATA LOADER FOR: %s' %(h5py_file))
      logger.info('BATCH_SIZE: %i  SHUFFLE: %i  MODE: %s' %(batch_size, shuffle, mode))
      logger.info('NUMBER OF DATA IS A MULTIPLE OF BATCH SIZE')

    if mode == 'train':
      dl_batch_size = batch_size
    elif mode == 'test':
      dl_batch_size = 1
    else:
      raise ValueError('Only train and test modes allowed')

    img0, img1, img2, uvd_norm_gt, _, _, _, _, = pd.read_data_h5py(h5py_file)
    num_data = img0.shape[0]
    # set number of data to be a multiple of batch size
    num_data_batch = batch_size*(img0.shape[0]//batch_size)
    img0 = img0[:num_data_batch]
    img1 = img1[:num_data_batch]
    img2 = img2[:num_data_batch]
    uvd_norm_gt = uvd_norm_gt[:num_data_batch]
    img0 = pd.move_channel_dim_3_to_1(img0).astype('float32')
    img1 = pd.move_channel_dim_3_to_1(img1).astype('float32')
    img2 = pd.move_channel_dim_3_to_1(img2).astype('float32')
    uvd_norm_gt = np.reshape(uvd_norm_gt, (-1, 63))
    fpha_dataset = FPHA_dataset(img0, img1, img2, uvd_norm_gt)

    if logger:
      logger.info('DATA SIZE: %i' %(img0.shape[0]))

    return torch.utils.data.DataLoader(dataset=fpha_dataset,
                                       batch_size=dl_batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)
