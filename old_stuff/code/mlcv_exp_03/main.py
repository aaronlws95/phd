import torch
import argparse

from src import ROOT, parse
from src.models import get_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--cfg', type=str, required=True)
    p.add_argument('-m','--mode', type=str, choices=['train','test'], required=True)
    p.add_argument('-e','--epoch', type=str, default=0)
    p.add_argument('-s','--split', type=str, default=None)
    args = p.parse_args()

    cfg         = parse(args.cfg)
    mode        = args.mode
    load_epoch  = int(args.epoch) if args.epoch != 'best' else args.epoch
    gpu_id      = [int(i) for i in cfg['device'].split(',')]

    cfg['device']       = gpu_id
    cfg['mode']         = mode
    cfg['load_epoch']   = load_epoch

    torch.cuda.set_device(gpu_id[0])
    torch.backends.cudnn.benchmark = True

    model = get_model(cfg)
    if len(gpu_id) > 1:
        model.net = torch.nn.DataParallel(model.net, device_ids=gpu_id)

    if mode == 'train':
        model.train()
    else:
        if args.split:
            split = map(str, args.split.strip('[]').split(','))
        else:
            split = ['train', 'test']
        cfg['split'] = split
        model.predict(cfg)

if __name__ == '__main__':
    main()