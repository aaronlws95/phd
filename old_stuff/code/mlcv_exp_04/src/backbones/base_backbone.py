import torch

class Base_Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels    = None
        self.out_filters    = None
        self.stride         = None

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False