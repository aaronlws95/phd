import torch

from src.datasets.fpha_hand_dataset import FPHA_Hand_Dataset
from src.datasets.ho3d_hand_dataset  import HO3D_Hand_Dataset
from src.datasets.ho3d_hand_object_dataset  import HO3D_Hand_Object_Dataset
from src.datasets.fpha_bbox_dataset import FPHA_Bbox_Dataset
from src.datasets.egtea_bbox_dataset import EGTEA_Bbox_Dataset
from src.datasets.egohands_bbox_dataset import Egohands_Bbox_Dataset
from src.datasets.eyth_bbox_dataset import EYTH_Bbox_Dataset
from src.datasets.egodexter_bbox_dataset import EgoDexter_Bbox_Dataset
from src.datasets.concatdata_bbox_dataset import ConcatData_Bbox_Dataset
from src.datasets.fpha_hand_crop_class_dataset import FPHA_Hand_Crop_Class_Dataset
datasets = {}
datasets['fpha_hand'] = FPHA_Hand_Dataset
datasets['ho3d_hand'] = HO3D_Hand_Dataset
datasets['ho3d_hand_object'] = HO3D_Hand_Object_Dataset
datasets['fpha_bbox'] = FPHA_Bbox_Dataset
datasets['egtea_bbox'] = EGTEA_Bbox_Dataset
datasets['egohands_bbox'] = Egohands_Bbox_Dataset
datasets['egodexter_bbox'] = EgoDexter_Bbox_Dataset
datasets['eyth_bbox'] = EYTH_Bbox_Dataset
datasets['concatdata_bbox'] = ConcatData_Bbox_Dataset
datasets['fpha_hand_crop_class'] = FPHA_Hand_Crop_Class_Dataset

def get_dataset(cfg, split):
    split_types = ['train', 'test', 'val']
    if split not in split_types:
        raise ValueError('Invalid split type {}'.format(split))

    dataset_name = cfg['dataset']
    dataset = datasets[dataset_name](cfg, split)
    return dataset

def get_dataloader(cfg, dataset, add_kwargs=None):
    kwargs = {
        'batch_size'    : int(cfg['batch_size']),
        'shuffle'       : cfg['shuffle'],
        'num_workers'   : int(cfg['num_workers']),
        'sampler'       : None,
        'pin_memory'    : True
    }

    if add_kwargs is not None:
        for k, v in add_kwargs.item():
            kwargs[k] = v

    return torch.utils.data.DataLoader(dataset, **kwargs)