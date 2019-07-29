import torch.utils.data
import os
import sys
import numpy as np
import pickle
import lmdb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.prepare_data as pd
import dataset.FPHA_dataset as fpha
from utils.logger import get_logger
from utils.directory import DATA_DIR

def znb_pose_net_fpha_dataloader(save_prefix, dataset, batch_size=8, shuffle=False, num_workers=12, logger=None, sampler=None):
    if logger:
      logger.info('CREATING POSE NET DATA LOADER FOR: %s' %save_prefix)
      logger.info('BATCH_SIZE: %i  SHUFFLE: %i' %(batch_size, shuffle))
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       sampler=sampler)

def znb_pred_smap_fpha_dataloader(save_prefix, dataset, batch_size=8, shuffle=False, num_workers=12, logger=None, sampler=None):
    if logger:
      logger.info('CREATING PREDICTED SMAP DATA LOADER FOR: %s' %save_prefix)
      logger.info('BATCH_SIZE: %i  SHUFFLE: %i' %(batch_size, shuffle))
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       sampler=sampler)

def znb_lift_net_fpha_dataloader(save_prefix, dataset, batch_size=8, shuffle=False, num_workers=12, logger=None, sampler=None):
    if logger:
      logger.info('CREATING LIFT NET DATA LOADER FOR: %s' %save_prefix)
      logger.info('BATCH_SIZE: %i  SHUFFLE: %i' %(batch_size, shuffle))
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       sampler=sampler)
