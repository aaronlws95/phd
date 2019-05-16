import torch
import torch.nn as nn
from torchvision.transforms import Resize
import numpy as np
import sys
import os
import argparse
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from utils.directory import DATA_DIR
from utils.logger import TBLogger
import model as m
import loss as l
import dataset as d
import hpo_net as hnet

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(0)
# arguments
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2345 train_pose_net.py --exp base_pose_net
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--map_location', type=str, default=None)
    args = parser.parse_args()

    # setup logging
    tb_logger = TBLogger(os.path.join(DATA_DIR, args.exp, 'log'))

    def init_dist(backend='nccl'):
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend)

    if args.gpu_id is None:
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        torch.cuda.set_device(args.gpu_id)
        rank = -1

    # if mp.get_start_method(allow_none=True) != 'spawn' and args.num_workers>0:
        # mp.set_start_method('spawn')
        # if rank<=0:
            # print('SET MULTIPROCESSING TO SPAWN')

    if rank<=0:
        if args.gpu_id is None:
            print('DISTRIBUTED TRAINING')
        else:
            print('SINGLE GPU TRAINING: %i' %(args.gpu_id))

    CKPT_DIR = os.path.join(DATA_DIR, args.exp, 'ckpt')
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    # load data
    init_epoch = 0
    if args.epoch != 0:
        load_dir = os.path.join(CKPT_DIR, 'model_%i.state' %args.epoch)
        if args.map_location:
            ckpt = torch.load(load_dir, map_location=args.map_location)
        else:
            ckpt = torch.load(load_dir)
        init_epoch = ckpt['epoch']+1
        print('LOADED CHECKPOINT: %i' %(init_epoch-1))
    else:
        ckpt = None

    net = hnet.HPO_Net()
    weightfile = os.path.join(DATA_DIR, 'yolov2.weights')
    # if init_epoch == 0:
        # print('LOADED yolov2 WEIGHTS')
        # net.darknet.load_weights(weightfile)
    # net.darknet.print_network()
    loss = l.HPOLoss()
    lr = 0.0001
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    model = m.Model(net, loss, optimizer, scheduler, gpu_id=args.gpu_id, ckpt=ckpt)

    save_prefix = 'train_fpha'
    aug = True
    dataset = d.FPHA(save_prefix, aug=aug)

    if args.gpu_id is None:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    shuffle = True
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=args.batch_size,
                                               shuffle=shuffle,
                                               num_workers=args.num_workers,
                                               sampler=sampler)
    print('CREATED TRAIN_LOADER')
    print('BATCH_SIZE: ', args.batch_size)
    print('SHUFFLE: ', shuffle)
    print('NUM_WORKERS: ', args.num_workers)
    print('AUGMENT: ', aug)

    save_prefix = 'test_fpha'
    val_dataset = d.FPHA(save_prefix)
    test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               sampler=None)
    print('CREATED TEST_LOADER')

    total_step = len(train_loader)
    cur_val_loss = 0
    time_elapsed = 0
    avg_step_time = 0
    for epoch in range(init_epoch, args.num_epochs):

        if args.gpu_id is None:
            train_sampler.set_epoch(epoch)

        if scheduler:
            model.scheduler.step()

        for param_group in model.optimizer.param_groups:
            cur_lr = param_group['lr']

        for cur_step, (img, uvd_gt_no_offset, uvd_gt, hand_cell_idx) in enumerate(train_loader):
            # train
            time_start = time.time()
            img = img.cuda()
            uvd_gt = uvd_gt.cuda()
            uvd_gt_no_offset = uvd_gt_no_offset.cuda()
            hand_cell_idx = hand_cell_idx.cuda()

            pred_uvd_no_offset, pred_uvd, pred_conf = model.net(img)
            loss = model.loss(pred_uvd_no_offset, uvd_gt_no_offset, pred_uvd, pred_conf, uvd_gt, hand_cell_idx)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # print(model.net.darknet.models[0][0].weight.grad)

            # log
            if (cur_step+1)%args.print_freq == 0 and rank <= 0:
                step_time = time.time() - time_start
                print('[epoch %i, step %i/%i, loss: %f, val_loss: %f, lr: %f, step_time/s: %f]' %(epoch+1, cur_step+1, total_step, loss.item(), cur_val_loss, cur_lr, step_time))

        if (epoch+1)%args.val_freq == 0 and rank <=0:
            print('VALIDATING EPOCH %i' %(epoch+1))

            # report validation loss
            model.net.eval()
            with torch.no_grad():
                val_loss = 0
                for i, (img, uvd_gt_no_offset, uvd_gt, hand_cell_idx) in enumerate(test_loader):
                    print('[validating epoch: %i img %i/%i]' %(epoch+1, i, len(test_loader)))
                    img = img.cuda()
                    uvd_gt = uvd_gt.cuda()
                    uvd_gt_no_offset = uvd_gt_no_offset.cuda()
                    hand_cell_idx = hand_cell_idx.cuda()
                    pred_uvd_no_offset, pred_uvd, pred_conf = model.net(img)
                    val_loss += model.loss(pred_uvd_no_offset, uvd_gt_no_offset, pred_uvd, pred_conf, uvd_gt, hand_cell_idx).item()
                val_loss /= len(test_loader)
                cur_val_loss = val_loss
            model.net.train()

        # tensorboard logging
        if rank <= 0:
            if (epoch+1)%args.val_freq == 0:
                tb_log_info = {'loss': loss.item(), 'val_loss' : cur_val_loss}
            else:
                tb_log_info = {'loss': loss.item()}
            for tag, value in tb_log_info.items():
                tb_logger.scalar_summary(tag, value, epoch+1)

            # save checkpoint
            if (epoch+1)%args.save_freq == 0:
                    print('SAVING CHECKPOINT EPOCH %i' %(epoch+1))
                    if isinstance(model.net, nn.parallel.DistributedDataParallel):
                        network = model.net.module
                    else:
                        network = model.net
                    model_state_dict = network.state_dict()
                    for key, param in model_state_dict.items():
                        model_state_dict[key] = param.cpu()
                    state = {'epoch': epoch, 'model_state_dict': model_state_dict, 'optimizer_state_dict': model.optimizer.state_dict(), 'loss': loss,}
                    if model.scheduler is not None:
                        state['scheduler_state_dict'] = model.scheduler.state_dict()
                    torch.save(state, os.path.join(CKPT_DIR, 'model_%i.state' %(epoch+1)))

if __name__ == '__main__':
    main()
