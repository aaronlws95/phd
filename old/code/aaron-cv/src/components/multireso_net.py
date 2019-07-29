import torch
import torch.nn     as nn

from src.components import Network

class Multireso_net(nn.Module):
    def __init__(self, net0_cfgfile, net1_cfgfile, 
                 net2_cfgfile, dense_cfgfile):
        super().__init__()
        self.net0   = Network(net0_cfgfile)
        self.net1   = Network(net1_cfgfile)
        self.net2   = Network(net2_cfgfile)
        self.dense  = Network(dense_cfgfile)

    def forward(self, x):
        x0, x1, x2 = x
        x0 = self.net0(x0)[0]
        x1 = self.net1(x1)[0]
        x2 = self.net2(x2)[0]
        out = torch.cat((x0,x1,x2), dim=1)
        out = out.reshape(out.shape[0], -1)
        out = self.dense(out)[0]
        out = out.view(out.shape[0], 21, 3)
        return out