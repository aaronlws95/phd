import torch

def get_optimizer(cfg, net):
    name = cfg['optimizer']
    if name == 'Adam':
        lr          = float(cfg['learning_rate'])
        decay       = float(cfg['decay']) #default=0
        return torch.optim.Adam(net.parameters(), lr=lr,
                                weight_decay=decay)
    elif name == 'SGD':
        lr          = float(cfg['learning_rate'])
        decay       = float(cfg['decay'])
        momentum    = float(cfg['momentum'])
        return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                               dampening=0, weight_decay=decay)
    elif name == "SGD_YOLO":
        """
        SGD as in https://github.com/marvis/pytorch-yolo2/blob/master/train.py
        divide by batch_size so the gradient doesn't blow up
        """
        batch_size  = int(cfg["batch_size"])
        lr          = float(cfg['learning_rate'])
        decay       = float(cfg['decay'])
        momentum    = float(cfg['momentum'])
        params_dict = dict(net.named_parameters())

        params = []
        for key, value in params_dict.items():
            if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                params += [{'params': [value],
                            'weight_decay': 0.0}]
            else:
                params += [{'params': [value],
                            'weight_decay': decay*batch_size}]
        return torch.optim.SGD(net.parameters(), lr=lr/batch_size,
                               momentum=momentum, dampening=0,
                               weight_decay=decay*batch_size)
    else:
        raise Exception(f'{name} is not a valid optimizer')