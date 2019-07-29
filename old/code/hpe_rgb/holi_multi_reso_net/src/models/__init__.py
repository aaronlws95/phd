import torch.optim
import torch.nn as nn

import models.multireso_net as mnet

class Base_Model():
    def __init__(self, device, net, loss, optimizer, gpu_ids=[0]):

        if len(gpu_ids) > 1:
            net = nn.DataParallel(net, device_ids=gpu_ids)

        self.net = net.to(device)
        self.loss = loss.to(device)
        self.optimizer = optimizer

    def load_ckpt(self, ckpt):
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

def get_multireso_model(device, lr=0.0003, logger=None):
    if logger:
        logger.info('LOAD MODEL: MULTIRESO_NET')
        logger.info('LEARNING RATE: %f' %lr)
        logger.info('OPTIMIZER: ADAM')
        logger.info('LOSS: MSE')

    net = mnet.Multireso_Net([64, 96, 128])
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return Base_Model(device, net, loss, optimizer)

def get_multireso_batchnorm_model(device, lr=0.0003, logger=None, gpu_ids=[0]):
    if logger:
            logger.info('LOAD MODEL: MULTIRESO_NET_BATCHNORM')
            logger.info('LEARNING RATE: %f' %lr)
            logger.info('OPTIMIZER: ADAM')
            logger.info('LOSS: MSE')

    net = mnet.Multireso_Net_Batchnorm([64, 96, 128])
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return Base_Model(device, net, loss, optimizer, gpu_ids=gpu_ids)

def get_multireso_batchnorm_model_6144_3072(device, lr=0.0003, logger=None, gpu_ids=[0]):
    if logger:
            logger.info('LOAD MODEL: MULTIRESO_NET_BATCHNORM 6144 3072')
            logger.info('LEARNING RATE: %f' %lr)
            logger.info('OPTIMIZER: ADAM')
            logger.info('LOSS: MSE')

    net = mnet.Multireso_Net_Batchnorm_6144_3072([64, 96, 128])
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return Base_Model(device, net, loss, optimizer, gpu_ids=gpu_ids)


