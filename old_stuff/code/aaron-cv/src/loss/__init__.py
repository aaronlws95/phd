from src.loss.region_noclass_1bbox  import RegionLoss_NoClass_1Bbox 
from src.loss.region                import RegionLoss
from src.loss.hpo                   import HPOLoss
from src.loss.hpo_action_noun       import HPO_Action_NounLoss
from src.loss.ek_action_noun        import EK_Action_NounLoss

def get_loss(loss, cfg):
    if      loss == 'region_noclass_1bbox':
        return RegionLoss_NoClass_1Bbox(cfg)
    elif    loss == 'region':
        return RegionLoss(cfg)
    elif    loss == 'hpo':
        return HPOLoss(cfg)
    elif    loss == 'hpo_action_noun':
        return HPO_Action_NounLoss(cfg)
    elif    loss == 'ek_action_noun':
        return EK_Action_NounLoss(cfg)
    else:
        raise ValueError(f"{name} is not a valid loss")