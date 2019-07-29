import torch

def get_scheduler(cfg, optimizer):
    name = cfg["scheduler"]
    if name == "multistep":
        milestones  = [int(i) for i in cfg["milestones"].split(',')]
        gamma       = float(cfg["gamma"])
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=milestones, 
                                                    gamma=gamma)
    elif name == "reducelronplateau":
        factor      = float(cfg['factor'])
        patience    = int(cfg['patience'])
        cooldown    = int(cfg['cooldown'])
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=factor, 
                                                           patience=patience, 
                                                           verbose=True, 
                                                           threshold=0.0001, 
                                                           threshold_mode='rel', 
                                                           cooldown=cooldown, 
                                                           min_lr=0, 
                                                           eps=1e-08)            
    elif name == None:
        return None
    else:
        raise ValueError(f"{name} is not a valid scheduler")