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

from utils.logger import Logger
from utils.json_utils import parse
from models import get_model
from datasets import get_dataloader, get_dataset

def setup_device(device):
    if device == "cpu":
        rank = -1
    else:
        gpu_id = device
        torch.cuda.set_device(gpu_id)
        # rank <= 0 to ensure only one system does logging, saving, etc
        rank = -1
    return rank

def data_load_to_cuda(data_load):
    data_load_cuda = []
    for x in data_load:
        if isinstance(x, torch.Tensor):
            data_load_cuda.append(x.cuda())
        else:
            data_load_cuda.append(x)
    return data_load_cuda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--enter", type=int, choices=[0,1], default=1)
    args = parser.parse_args()
    
    conf            = parse(args.conf)
    train_mode        = False
    device          = conf["device"]
    load_epoch      = args.epoch if args.epoch else conf["load_epoch"]
    exp_dir         = conf["exp_dir"]
    rank            = setup_device(device)
    logger          = Logger(conf["logger"], rank)

    if load_epoch == 0 and args.weights is None:
        raise ValueError("Cannot predict for epoch 0.")   

    if device == "distributed":
        raise ValueError("Cannot use distributed for prediction")   

    # logging
    logger.log("READ CONFIG: {}".format(conf["exp_dir"]))
    if device == "cpu":
        logger.log("CPU PREDICTION") # not tested
    else:
        logger.log(f"SINGLE GPU PREDICTION: {device}")
        
    if args.weights:
        load_epoch = 0
        conf["model"]["pretrain"] = args.weights

    # get model
    model = get_model(conf["model"], 
                      device, 
                      load_epoch, 
                      train_mode,
                      exp_dir,
                      conf["deterministic"],
                      logger)

    if args.split:
        split = map(str, args.split.strip('[]').split(','))
    else:
        split = ["train", "test"]

    if args.enter:
        input("Press Enter to begin predicting...")

    for data_split in split:
        logger.log("PREDICTING IN DATA SPLIT: {}".format(data_split.upper()))
        dataset = get_dataset(conf["dataset"][data_split], 
                              train_mode, 
                              model, 
                              conf["deterministic"])
        data_loader = get_dataloader(conf["dataset"][data_split],
                                     dataset,
                                     None,
                                     device, 
                                     train_mode,
                                     logger=logger)
        # predict
        model.net.eval()
        with torch.no_grad():
            for data_load in tqdm(data_loader):
                if device != "cpu":
                    data_load = data_load_to_cuda(data_load)
                model.predict_step(data_load)
            model.save_predictions(data_split)

if __name__ == "__main__":
    main()
