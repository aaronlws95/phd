from torch.utils.data                       import DataLoader
from src.datasets.fpha_bbox_hand            import FPHA_Bbox_Hand
from src.datasets.fpha_bbox_hand_flip       import FPHA_Bbox_Hand_Flip
from src.datasets.fpha_bbox                 import FPHA_Bbox
from src.datasets.voc_bbox                  import VOC_Bbox
from src.datasets.fpha_hand                 import FPHA_Hand
from src.datasets.fpha_hand_flip            import FPHA_Hand_Flip
from src.datasets.rhd_smap                  import RHD_Smap
from src.datasets.rhd_smap_canon            import RHD_Smap_Canon
from src.datasets.rhd_mask                  import RHD_Mask
from src.datasets.fpha_multireso_crop_hand  import FPHA_Multireso_Crop_Hand
from src.datasets.fpha_mr_crop_normuvd_hand import FPHA_MR_Crop_Normuvd_Hand
from src.datasets.ek_tsn_labels             import EK_TSN_Labels
from src.datasets.tsn_labels                import TSN_Labels
from src.datasets.fpha_paf                  import FPHA_PAF
from src.datasets.fpha_hand_action_noun     import FPHA_Hand_Action_Noun
from src.datasets.ek_hand_action_noun       import EK_Hand_Action_Noun

def get_dataset(cfg, kwargs={}):
    dataset = cfg['dataset']
    if      dataset == 'fpha_bbox_hand':
        return FPHA_Bbox_Hand(cfg, **kwargs)
    elif    dataset == 'fpha_bbox_hand_flip':
        return FPHA_Bbox_Hand_Flip(cfg, **kwargs)
    elif    dataset == 'fpha_bbox':
        return FPHA_Bbox(cfg, **kwargs)
    elif    dataset == 'voc_bbox':
        return VOC_Bbox(cfg, **kwargs)
    elif    dataset == 'fpha_hand':
        return FPHA_Hand(cfg, **kwargs)
    elif    dataset == 'fpha_hand_flip':
        return FPHA_Hand_Flip(cfg, **kwargs)
    elif    dataset == 'rhd_smap':
        return RHD_Smap(cfg, **kwargs)
    elif    dataset == 'rhd_smap_canon':
        return RHD_Smap_Canon(cfg, **kwargs)
    elif    dataset == 'rhd_mask':
        return RHD_Mask(cfg, **kwargs)
    elif    dataset == 'fpha_multireso_crop_hand':
        return FPHA_Multireso_Crop_Hand(cfg, **kwargs)
    elif    dataset == 'fpha_mr_crop_normuvd_hand':
        return FPHA_MR_Crop_Normuvd_Hand(cfg, **kwargs)
    elif    dataset == 'fpha_paf':
        return FPHA_PAF(cfg, **kwargs)
    elif    dataset == 'fpha_hand_action_noun':
        return FPHA_Hand_Action_Noun(cfg, **kwargs)
    elif    dataset == 'ek_hand_action_noun':
        return EK_Hand_Action_Noun(cfg, **kwargs)
    elif    dataset == 'ek_tsn_labels':
        raise ValueError(f"Call {dataset} within model class")
    elif    dataset == 'tsn_labels':
        raise ValueError(f"Call {dataset} within model class")
    else:
        raise ValueError(f"{dataset} is not a valid dataset")

def get_dataloader(dataset, sampler, kwargs):
    data_loader = DataLoader(dataset=dataset,
                             sampler=sampler,
                             **kwargs)
    return data_loader