import torch
import torch.nn                                             as nn
import numpy                                                as np
import os
import argparse
import time
import torch.distributed                                    as dist
from torch.nn.parallel  import DistributedDataParallel
import logging
from pathlib            import Path

from src.models         import get_model
from src.utils          import TBLogger, Logger, parse_data_cfg, DATA_DIR

def setup_device(device):
    if device == "distributed":
        def init_dist(backend="nccl"):
            rank = int(os.environ["RANK"])
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(rank%num_gpus)
            dist.init_process_group(backend=backend)
            
        init_dist()
        rank = torch.distributed.get_rank()
    else:
        gpu_id = int(device)
        torch.cuda.set_device(gpu_id)
        # rank <= 0 to ensure only one system does logging, saving, etc
        rank = -1
    return rank

# To run distributed training:
# CUDA_VISIBLE_DEVICES=0,1 python -m 
# torch.distributed.launch --nproc_per_node=2 --master_port=2345 train.py
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--mode", type=str, choices=['train','test'], required=True)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--split", type=str, default=None)
    args = p.parse_args()
    
    torch.backends.cudnn.benchmark = True
    
    cfg             = parse_data_cfg(args.cfg)
    load_epoch      = args.epoch if args.epoch else 0
    rank            = setup_device(cfg['device']) 
    training        = args.mode == 'train'
    tb_logger       = TBLogger(Path(DATA_DIR)/cfg['exp_dir']/'log')
    logger          = Logger(rank)    
    
    model           = get_model(cfg, training, load_epoch, logger, tb_logger)

    # if model.device == "distributed":
    #     model.net = \
    #         DistributedDataParallel(model.net, 
    #                                 [torch.cuda.current_device()])     
    
    if training:
        model.train(rank)
    else:
        if args.split:
            split = map(str, args.split.strip('[]').split(','))
        else:
            split = ["train", "test"]
        model.predict(cfg, split)

if __name__ == "__main__":
    main()
