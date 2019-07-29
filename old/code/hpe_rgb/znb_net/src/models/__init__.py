import torch.optim
import torch.nn as nn
import torch

import models.znb_net as mnet

class Base_Model():
    def __init__(self, net, loss, optimizer, scheduler=None, gpu_id=None, ckpt=None):

        if ckpt:
            print('LOADED CHECKPOINT')
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

def get_pose_net(lr=0.0001, logger=None, gpu_id=None, ckpt=None):
    if logger:
        logger.info('LOAD MODEL: ZNB POSE NET')
        logger.info('LEARNING RATE: %f' %lr)
        logger.info('OPTIMIZER: ADAM')
        logger.info('LOSS: SCOREMAP LOSS')

    def scoremap_loss(pred, true):
        def l2dist(true, pred):
            return torch.mean(torch.sqrt(torch.mean((true-pred)**2, dim=(2,3))))
        # criterion = nn.MSELoss()
        criterion = l2dist
        loss = (criterion(pred[0], true) + criterion(pred[1], true) + criterion(pred[2], true))
        return loss

    net = mnet.ZNB_Pose_Net()
    loss = scoremap_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    return Base_Model(net, loss, optimizer, scheduler, gpu_id, ckpt=ckpt)

def get_CPM(lr=0.0001, logger=None, gpu_id=None, ckpt=None):
    if logger:
        logger.info('LOAD MODEL: CPM')
        logger.info('LEARNING RATE: %f' %lr)
        logger.info('OPTIMIZER: ADAM')
        logger.info('LOSS: SCOREMAP LOSS')

    def scoremap_loss(pred, true):
        criterion = nn.MSELoss()
        loss = (criterion(pred[0], true) + criterion(pred[1], true) + criterion(pred[2], true) + criterion(pred[3], true) + criterion(pred[4], true) + criterion(pred[5], true))
        return loss

    net = mnet.CPM(k=21)
    loss = scoremap_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    return Base_Model(net, loss, optimizer, scheduler, gpu_id, ckpt=ckpt)

def get_lifting_net(lr=0.00001, logger=None, gpu_id=None, ckpt=None):
    if logger:
        logger.info('LOAD MODEL: ZNB LIFTING NET')
        logger.info('LEARNING RATE: %f' %lr)
        logger.info('OPTIMIZER: ADAM')
        logger.info('LOSS: MSE')

    def lifting_loss(pred_rot_mat, pred_xyz_canon, rot_mat, xyz_gt_canon):
        criterion = nn.MSELoss()
        loss_xyz_canon = criterion(pred_xyz_canon, xyz_gt_canon)
        loss_rot_mat = criterion(pred_rot_mat, rot_mat)
        total_loss = loss_xyz_canon + loss_rot_mat
        return total_loss

    net = mnet.ZNB_Lift_Net()
    loss = lifting_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.1)
    return Base_Model(net, loss, optimizer, scheduler, gpu_id=gpu_id, ckpt=ckpt)


