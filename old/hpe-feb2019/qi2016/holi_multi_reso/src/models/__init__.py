import torch.optim
import torch.nn as nn

from .multireso_net import Multireso_Net

class Base_Model():
    def __init__(self, device, net, loss, optimizer):
        self.net = net.to(device)
        self.loss = loss.to(device)
        self.optimizer = optimizer

    def load_ckpt(self, ckpt):
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

def get_multireso_model(device, lr=0.0003):
    net = Multireso_Net([64, 96, 128])
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return Base_Model(device, net, loss, optimizer)
