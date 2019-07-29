import torch

def get_scheduler(cfg, optimizer):
    name = cfg['scheduler']
    if name == 'multistep':
        milestones  = [int(i) for i in cfg['milestones'].split(',')]
        gamma       = float(cfg['gamma'])
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=milestones, 
                                                    gamma=gamma)
    elif name == 'reduce_lr_on_plateau':
        patience = int(cfg['patience'])
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min',
                                                           patience=patience)
    else:
        raise Exception('{} is not a valid scheduler'.format(name))