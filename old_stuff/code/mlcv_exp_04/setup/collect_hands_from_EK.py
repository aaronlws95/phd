import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT, parse
from src.models import get_model
from src.datasets.transforms import *
from src.datasets import get_dataset, get_dataloader
from src.utils import *

p = argparse.ArgumentParser()
p.add_argument('-t', '--thresh', type=float, required=True)
args = p.parse_args()

cfg_dir = 'mlcv-exp/data/cfg/'

dataset ='concatdata'
model_name = 'yolov2_bbox'
exp = 'exp1'
epoch = 200

cfg_name = '{}_{}_{}.cfg'.format(dataset, model_name, exp)
cfg = parse(str(Path(ROOT)/cfg_dir/model_name/cfg_name))
cfg['device'] = '1'
cfg['aug'] = None
cfg['batch_size'] = 1
cfg['shuffle'] = 1
cfg['mode'] = 'test'
cfg['load_epoch'] = epoch
model = get_model(cfg)
model.net.eval()

dataset_file = 'EPIC_KITCHENS_2018'
dts_folder = 'EK_frames'
thresh = args.thresh
print('thresh:', thresh)
model.collect_hands(Path(ROOT)/'datasets'/dataset_file/dts_folder, model_info='{}_{}_{}_{}_{}_{}'.format(dataset_file, dataset, model_name, exp, epoch, thresh), thresh=thresh)