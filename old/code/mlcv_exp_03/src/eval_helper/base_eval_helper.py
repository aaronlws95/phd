from pathlib import Path

from src import ROOT
from src.datasets import get_dataloader, get_dataset


class Base_Eval_Helper:
    def __init__(self, cfg, epoch, split):
        self.exp_dir    = Path(ROOT)/'mlcv-exp'/'data'/'exp'/cfg['model']/cfg['exp']
        self.split      = split
        self.dataset    = get_dataset(cfg, self.split)
        self.dataloader = get_dataloader(cfg, self.dataset)
        self.epoch      = epoch
        self.gt         = self.dataset.get_gt()
        self.data_load  = next(iter(self.dataloader))

    def get_len(self):
        raise NotImplementedError()

    def get_pred(self):
        raise NotImplementedError()

    def visualize_one_prediction(self, idx):
        raise NotImplementedError()

    def visualize_dataloader(self, idx):
        self.dataset.visualize(self.data_load, idx)

    def visualize_multi_dataloader(self):
        self.dataset.visualize_multi(self.data_load)

    def eval(self):
        raise NotImplementedError()