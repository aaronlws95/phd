import torch

class Reorg(torch.nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        s = self.stride
        assert(x.dim() == 4)
        B, C, H, W = x.size()
        assert(H%s == 0)
        assert(W%s == 0)
        x = x.view(B, C, H//s, s, W//s, s).transpose(3,4).contiguous()
        x = x.view(B, C, (H//s)*(W//s), s*s).transpose(2,3).contiguous()
        x = x.view(B, C, s*s, H//s, W//s).transpose(1,2).contiguous()
        x = x.view(B, C*s*s, H//s, W//s)
        return x