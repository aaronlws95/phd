import torch

from src.networks.modules import *
from src.networks.backbones.yolov2 import YOLOV2

class Multireso_Hand_Crop_Net(torch.nn.Module):
  def __init__(self, num_filter=[64, 96, 128], channels=3):
    super(Multireso_Hand_Crop_Net, self).__init__()
    self.conv_block_0 = self._conv_block(num_filter, [5, 4, 3], [4, 2, 2], channels)
    self.conv_block_1 = self._conv_block(num_filter, [5, 3, 3], [2, 2, 2], channels)
    self.conv_block_2 = self._conv_block(num_filter, [5, 5, 3], [2, 1, 1], channels)
    self.dense_block = self._dense_block(6144)

  def _conv_block(self, num_filter, k, p, channels):
    def conv_and_maxpool(num_filter, k, p, channels):
      conv = torch.nn.Conv2d(channels, num_filter, kernel_size=k)
      relu = torch.nn.ReLU()
      maxpool = torch.nn.MaxPool2d(kernel_size=p)
      return torch.nn.Sequential(conv, relu, maxpool)

    c0 = conv_and_maxpool(num_filter[0], k[0], p[0], channels)
    c1 = conv_and_maxpool(num_filter[1], k[1], p[1], num_filter[0])
    c2 = conv_and_maxpool(num_filter[2], k[2], p[2], num_filter[1])
    return torch.nn.Sequential(c0, c1, c2)

  def _dense_block(self, in_features):
    d0 = torch.nn.Linear(in_features, in_features//2)
    relu0 = torch.nn.ReLU()
    d1  = torch.nn.Linear(in_features//2, in_features//4)
    relu1 = torch.nn.ReLU()
    d2 = torch.nn.Linear(in_features//4, 63)
    return torch.nn.Sequential(d0, relu0, d1, relu1, d2)

  def forward(self, x):
      x0, x1, x2 = x
      x0 = self.conv_block_0(x0)
      x1 = self.conv_block_1(x1)
      x2 = self.conv_block_2(x2)
      out = torch.cat((x0,x1,x2), dim=1)
      out = out.reshape(out.shape[0], -1)
      out = self.dense_block(out)
      return out

class Multireso_Hand_Crop_BN_Net(torch.nn.Module):
  def __init__(self, num_filter=[64, 96, 128], channels=3):
    super(Multireso_Hand_Crop_BN_Net, self).__init__()
    self.conv_block_0 = self._conv_block(num_filter, [5, 4, 3], [4, 2, 2], channels)
    self.conv_block_1 = self._conv_block(num_filter, [5, 3, 3], [2, 2, 2], channels)
    self.conv_block_2 = self._conv_block(num_filter, [5, 5, 3], [2, 1, 1], channels)
    self.dense_block = self._dense_block(6144)

  def _conv_block(self, num_filter, k, p, channels):
    def conv_and_maxpool(num_filter, k, p, channels):
      conv = torch.nn.Conv2d(channels, num_filter, kernel_size=k)
      batchnorm = torch.nn.BatchNorm2d(num_features=num_filter)
      relu = torch.nn.ReLU()
      maxpool = torch.nn.MaxPool2d(kernel_size=p)
      return torch.nn.Sequential(conv, batchnorm, relu, maxpool)

    c0 = conv_and_maxpool(num_filter[0], k[0], p[0], channels)
    c1 = conv_and_maxpool(num_filter[1], k[1], p[1], num_filter[0])
    c2 = conv_and_maxpool(num_filter[2], k[2], p[2], num_filter[1])
    return torch.nn.Sequential(c0, c1, c2)

  def _dense_block(self, in_features):
    batchnorm_prev = torch.nn.BatchNorm1d(num_features=in_features)
    d0 = torch.nn.Linear(in_features, in_features//2)
    batchnorm0 = torch.nn.BatchNorm1d(num_features=in_features//2)
    relu0 = torch.nn.ReLU()
    d1  = torch.nn.Linear(in_features//2, in_features//4)
    batchnorm1 = torch.nn.BatchNorm1d(num_features=in_features//4)
    relu1 = torch.nn.ReLU()
    d2 = torch.nn.Linear(in_features//4, 63)
    return torch.nn.Sequential(batchnorm_prev, d0, batchnorm0, relu0, d1, batchnorm1, relu1, d2)

  def forward(self, x):
      x0, x1, x2 = x
      x0 = self.conv_block_0(x0)
      x1 = self.conv_block_1(x1)
      x2 = self.conv_block_2(x2)
      out = torch.cat((x0,x1,x2), dim=1)
      out = out.reshape(out.shape[0], -1)
      out = self.dense_block(out)
      return out