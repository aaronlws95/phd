import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src import ROOT
from src.eval_helper.base_eval_helper import Base_Eval_Helper
from src.utils import *

class Hand_Crop_Disc_Helper(Base_Eval_Helper):
    def __init__(self, cfg, epoch, split):
        super().__init__(cfg, epoch, split)
        self.img_root = Path(ROOT)/cfg['img_root']
        self.img_rsz = int(cfg['img_rsz'])
        self.img_paths = self.gt

    def get_len(self):
        return len(self.img_paths)

    def get_pred(self):
        pass

    def visualize_one_prediction(self, idx):
        pass

    def eval(self):
        pass