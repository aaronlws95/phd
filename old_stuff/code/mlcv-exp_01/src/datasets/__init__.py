import torch

def get_dataset(cfg, split):
    split_types = ['train', 'test', 'val']
    if split not in split_types:
        raise ValueError('Invalid split type {}'.format(split))

    dataset = cfg['dataset']
    if dataset == 'fpha_ar_seq':
        from src.datasets.fpha_ar_seq_dataset import FPHA_AR_Seq_Dataset
        return FPHA_AR_Seq_Dataset(cfg, split)
    elif dataset == 'fpha_bbox':
        from src.datasets.fpha_bbox_dataset import FPHA_Bbox_Dataset
        return FPHA_Bbox_Dataset(cfg, split)
    elif dataset == 'ek_ar':
        from src.datasets.ek_ar_dataset import EK_AR_Dataset
        return EK_AR_Dataset(cfg, split)
    elif dataset == 'fpha_ar':
        from src.datasets.fpha_ar_dataset import FPHA_AR_Dataset
        return FPHA_AR_Dataset(cfg, split)
    elif dataset == 'fpha_hand':
        from src.datasets.fpha_hand_dataset import FPHA_Hand_Dataset
        return FPHA_Hand_Dataset(cfg, split)
    elif dataset == 'fpha_bbox_ar':
        from src.datasets.fpha_bbox_ar_dataset import FPHA_Bbox_AR_Dataset
        return FPHA_Bbox_AR_Dataset(cfg, split)
    else:
        raise Exception('{} is not a valid dataset'.format(dataset))

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