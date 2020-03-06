import torch

from src.networks.modules.reorg import Reorg
from src.networks.modules.convlstm import ConvLSTM, ConvLSTMCell

def _get_activation(activation):
    if activation == 'leaky':
        return torch.nn.LeakyReLU(0.1)
    elif activation == 'relu':
        return torch.nn.ReLU()

def conv(in_c, out_c, k_size, stride, pad, bias, bn, activation):
    modules = torch.nn.Sequential()
    pad = (k_size - 1)//2 if pad else 0
    modules.add_module('conv', torch.nn.Conv2d(in_channels=in_c, 
                                               out_channels=out_c,
                                               kernel_size=k_size,
                                               stride=stride,
                                               padding=pad,
                                               bias=bias))
    if bn:
        modules.add_module('bn', torch.nn.BatchNorm2d(out_c))
    if activation:
        modules.add_module(activation, _get_activation(activation))
    return modules