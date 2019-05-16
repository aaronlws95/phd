import torch
import torch.nn as nn
from torchvision.transforms import Resize
import numpy as np
import sys
import os
import argparse
import time
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import logging
from tqdm import tqdm

from utils.logger import TBLogger, Logger
from utils.json_utils import parse
from models import get_model
from datasets import get_dataloader, get_dataset

def setup_device(device):
    if device == "distributed":
        def init_dist(backend="nccl"):
            rank = int(os.environ["RANK"])
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend=backend)
            
        init_dist()
        rank = torch.distributed.get_rank()
    elif device == "cpu":
        rank = -1
    else:
        gpu_id = device
        torch.cuda.set_device(gpu_id)
        # rank <= 0 to ensure only one system does logging, saving, etc
        rank = -1
    return rank

def validate(model, val_loader, epoch, device):
    print(f"VALIDATING EPOCH {epoch}")
    model.net.eval()
    with torch.no_grad():
        for data_load in tqdm(val_loader):
            if device != "cpu":
                data_load = data_load_to_cuda(data_load)     
            model.valid_step(data_load)
        val_loss = model.get_valid_loss()
    model.net.train()
    return val_loss

def data_load_to_cuda(data_load):
    data_load_cuda = []
    for x in data_load:
        if isinstance(x, torch.Tensor):
            data_load_cuda.append(x.cuda())
        else:
            data_load_cuda.append(x)
    return data_load_cuda

def get_train_loader(conf, device, train_mode, model, logger):
    train_dataset = get_dataset(conf["dataset"]["train"], 
                                train_mode, 
                                model, 
                                conf["deterministic"])
    if device == "distributed":
        train_sampler = DistributedSampler(train_dataset)
        # distributed sampler deterministically shuffles based on epoch
        conf["dataset"]["train"]["shuffle"] = False
        if "actual_shuffle" in conf:
            conf["dataset"]["train"]["actual_shuffle"] = False
    else:
        train_sampler = None
    train_loader = get_dataloader(conf["dataset"]["train"], 
                                  train_dataset, 
                                  train_sampler, 
                                  device, 
                                  train_mode, 
                                  deterministic=conf["deterministic"],
                                  logger=logger)    
    return train_loader, train_sampler

def get_val_loader(conf, device, train_mode, logger):
    val_dataset = get_dataset(conf["dataset"]["val"], 
                                train_mode,
                                None, 
                                conf["deterministic"])
    val_loader = get_dataloader(conf["dataset"]["val"], 
                                val_dataset, 
                                None, 
                                device, 
                                train_mode, 
                                deterministic=conf["deterministic"],
                                logger=logger)
    return val_loader

# To run distributed training:
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2345 train.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--enter", type=int, choices=[0,1], default=1)
    args = parser.parse_args()
    
    conf            = parse(args.conf)
    train_mode      = True
    do_val          = conf["val_freq"] != 0
    device          = conf["device"]
    exp_dir         = conf["exp_dir"]
    load_epoch      = args.epoch if args.epoch else conf["load_epoch"]
    rank            = setup_device(device) # setup device
    
    # setup logging
    tb_logger       = TBLogger(os.path.join(exp_dir, "log"))
    logger          = Logger(conf["logger"], rank)    
    
    # logging
    logger.log("READ CONFIG: {}".format(conf["exp_dir"]))
    if device == "distributed":
        logger.log("DISTRIBUTED TRAINING") # not tested
    elif device == "cpu":
        logger.log("CPU TRAINING") # not tested
    else:
        logger.log(f"SINGLE GPU TRAINING: {device}")
    if conf["deterministic"]:
        logger.log("DETERMINISTIC")
    else:
        logger.log("NOT DETERMINISTIC")

    # get model
    model = get_model(conf["model"], 
                      device, 
                      load_epoch, 
                      train_mode,
                      exp_dir,
                      conf["deterministic"],
                      logger)

    # torch.backends.cudnn.benchmark = True only if input size not changing
    torch.backends.cudnn.benchmark = True

    # training data loader
    train_loader, train_sampler = get_train_loader(conf, device, train_mode, model, logger)
    
    # validation data loader
    if do_val:
        val_loader = get_val_loader(conf, device, train_mode, logger)

    if device != "distributed" and args.enter:
        input("Press Enter to begin training...")

    # entire training loop
    for epoch in range(load_epoch, conf["max_epochs"]):
        
        if device == "distributed":
            train_sampler.set_epoch(epoch)
        if model.scheduler:
            model.scheduler.step()

        # initialise epoch
        model.init_epoch(epoch)
        
        # training epoch loop
        for cur_step, data_load in enumerate(train_loader):
            if device != "cpu":
                data_load = data_load_to_cuda(data_load)
            
            # train step
            loss = model.train_step(data_load)
            
            # get current learning rate
            for param_group in model.optimizer.param_groups:
                cur_lr = param_group["lr"]            
                
            # logging for each step
            if (cur_step+1)%logger.print_freq == 0 and rank <= 0:
                logger.log_dict["epoch"] = epoch+1
                logger.log_dict["step"] = "{}/{}".format(cur_step+1, len(train_loader))
                for key, val in loss.items():
                    logger.log_dict[key] = val
                logger.log_dict["lr"] = cur_lr
                logger.log_step(cur_step)

        # post-epoch processing
        if rank <= 0:
            # validation
            if do_val:
                if (epoch+1)%conf["val_freq"] == 0:
                    val_loss = validate(model, val_loader, epoch+1, device)
                    
            # tensorboard logging
            tb_logger.log_dict["loss"] = loss["loss"]   
            if do_val:
                if (epoch+1)%conf["val_freq"] == 0:
                    for key, val in val_loss.items():
                        tb_logger.log_dict[key] = val                    
            tb_logger.update_scalar_summary(epoch+1)
           
            # save checkpoint
            if (epoch+1)%model.save_freq == 0:
                model.save_ckpt(epoch+1)

if __name__ == "__main__":
    main()
