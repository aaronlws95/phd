import torch.optim
import torch.nn as nn
import torch

class Model():
    def __init__(self, net, loss, optimizer, scheduler=None, gpu_id=None, ckpt=None):

        if ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        if gpu_id != -1:
            net.cuda()
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            if gpu_id is None:
                net = nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])

        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = None
