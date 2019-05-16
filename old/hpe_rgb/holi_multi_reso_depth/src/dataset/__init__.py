import torch.utils.data
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.prepare_data as pd
from .FPHA_dataset import FPHA_dataset
from utils.logger import get_logger

def multireso_fpha_dataloader(h5py_file, batch_size=16, shuffle=True, num_workers=12, logger=None):
    if logger:
      logger.info('CREATING DATA LOADER FOR: %s' %(h5py_file))
      logger.info('BATCH_SIZE: %i  SHUFFLE: %i' %(batch_size, shuffle))

    img0, img1, img2, uvd_norm_gt, _, _, _, _, = pd.read_data_h5py(h5py_file)
    img0 = pd.move_channel_dim_3_to_1(img0)
    img1 = pd.move_channel_dim_3_to_1(img1)
    img2 = pd.move_channel_dim_3_to_1(img2)
    uvd_norm_gt = np.reshape(uvd_norm_gt, (-1, 63))
    fpha_dataset = FPHA_dataset(img0, img1, img2, uvd_norm_gt)
    return torch.utils.data.DataLoader(dataset=fpha_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)
