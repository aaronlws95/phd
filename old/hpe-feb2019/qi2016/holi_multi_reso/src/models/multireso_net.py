import torch
import torch.nn as nn

class Multireso_Net(nn.Module):
  def __init__(self, num_filter, channels=1):
    super(Multireso_Net, self).__init__()
    self.conv_block_0 = self._conv_block(num_filter, [5, 4, 3], [4, 2, 2], channels)
    self.conv_block_1 = self._conv_block(num_filter, [5, 3, 3], [2, 2, 2], channels)
    self.conv_block_2 = self._conv_block(num_filter, [5, 5, 3], [2, 1, 1], channels)
    self.dense_block = self._dense_block(6144)

  def _conv_block(self, num_filter, k, p, channels):
    def conv_and_maxpool(num_filter, k, p, channels):
      conv = nn.Conv2d(channels, num_filter, kernel_size=k)
      relu = nn.ReLU()
      maxpool = nn.MaxPool2d(kernel_size=p)
      return nn.Sequential(conv, relu, maxpool)

    c0 = conv_and_maxpool(num_filter[0], k[0], p[0], channels)
    c1 = conv_and_maxpool(num_filter[1], k[1], p[1], num_filter[0])
    c2 = conv_and_maxpool(num_filter[2], k[2], p[2], num_filter[1])
    return nn.Sequential(c0, c1, c2)

  def _dense_block(self, in_features):
    d0 = nn.Linear(in_features, in_features//2)
    relu0 = nn.ReLU()
    d1  = nn.Linear(in_features//2, in_features//4)
    relu1 = nn.ReLU()
    d2 = nn.Linear(in_features//4, 63)
    return nn.Sequential(d0, relu0, d1, relu1, d2)

  def forward(self, x0, x1, x2):
      x0 = self.conv_block_0(x0)
      x1 = self.conv_block_1(x1)
      x2 = self.conv_block_2(x2)
      out = torch.cat((x0,x1,x2), dim=1)
      out = out.reshape(out.shape[0], -1)
      out = self.dense_block(out)
      return out


