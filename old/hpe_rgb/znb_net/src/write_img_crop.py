import os
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.transform import resize
from utils.directory import DATASET_DIR
from utils.logger import get_logger
import utils.prepare_data as pd
logger = get_logger()

fpha_path = os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark')
video_path = os.path.join(fpha_path, 'Video_files')

# directories = [x[0] for x in os.walk(video_path)]
# for dr in directories:
#     new_dr = os.path.join(fpha_path, 'Video_files_256_crop', dr[53:])
#     os.mkdir(new_dr)
#     print('made', new_dr)

train_pairs, test_pairs = pd.get_fpha_data_list_general('color', DATASET_DIR)

logger.info('LOADED FPHA PAIRS')

file_name = [i for i,j in train_pairs]
xyz_gt = [j for i,j in train_pairs]
pd.write_data_to_file_img_crop(file_name, xyz_gt, logger)

file_name = [i for i,j in test_pairs]
xyz_gt = [j for i,j in test_pairs]
pd.write_data_to_file_img_crop(file_name, xyz_gt, logger)
