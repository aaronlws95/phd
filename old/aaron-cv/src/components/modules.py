import torch.nn as nn

def get_module(mdef, o_filters, idx):
    """ Get module given module_def 
    Args:
        mdef: dict. Module definitions
        o_filters: list. Previous filters
        idx: int. Module id
    Out:
        modules: Corresponding module
        filters: Current filters
    """
    modules = nn.Sequential()
    mtype = mdef['type']
    if mtype == 'convolutional':
        if 'batch_normalize' in mdef:
            bn = int(mdef['batch_normalize'])  
        else:
            bn = 0
            mdef['batch_normalize'] = 0
        filters = int(mdef['filters'])
        kernel_size = int(mdef['size'])
        pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
        conv2d = nn.Conv2d(in_channels=o_filters[-1],
                           out_channels=filters, 
                           kernel_size=kernel_size,
                           stride=int(mdef['stride']),
                           padding=pad,
                           bias=not bn)
        modules.add_module('conv_{}'.format(idx), conv2d)
        if bn:
            modules.add_module('batch_norm_{}'.format(idx), 
                               nn.BatchNorm2d(filters))
        mact = mdef['activation']
        if mact == 'leaky':
            modules.add_module('leaky_{}'.format(idx), 
                               nn.LeakyReLU(0.1, inplace=True))
        elif mact == 'relu':
               modules.add_module('relu_{}'.format(idx), 
                              nn.ReLU(inplace=True)) 
        elif mact == 'linear':  
               pass
        else:
            raise ValueError('Unknown activation type \'{}\''.format(mact))
    elif mtype == 'concat':
        layers = [int(x) for x in mdef['layers'].split(',')]
        filters = sum([o_filters[i + 1 if i > 0 else idx + i + 1] 
                       for i in layers])
        modules.add_module('concat_{}'.format(idx), EmptyModule())
    elif mtype == 'maxpool':
        kernel_size = int(mdef['size'])
        stride = int(mdef['stride'])
        filters = o_filters[-1]
        maxpool = nn.MaxPool2d(kernel_size=kernel_size, 
                               stride=stride, 
                               padding=int((kernel_size - 1)//2))
        modules.add_module('maxpool_{}'.format(idx), maxpool)
    elif mtype == 'avgpool':
        kernel_size = int(mdef['size'])
        stride = int(mdef['stride'])
        pad = int(mdef['pad'])
        filters = o_filters[-1]
        avgpool = nn.AvgPool2d(kernel_size=kernel_size,
                               stride=stride,
                               padding=pad)
        modules.add_module('avgpool_{}'.format(idx), avgpool)
    elif mtype == 'dropout':
        prob = float(mdef['prob']) # prob to zero an element
        filters = o_filters[-1]
        dropout = nn.Dropout(prob)
        modules.add_module('dropout_{}'.format(idx), dropout)        
    elif mtype == 'route':
        layers = [int(x) for x in mdef['layers'].split(',')]
        filters = sum([o_filters[i + 1 if i > 0 else idx + i + 1] 
                       for i in layers])
        modules.add_module('route_{}'.format(idx), EmptyModule())
    elif mtype == 'reorg':
        stride = int(mdef['stride'])
        filters = stride*stride*o_filters[-1]
        modules.add_module('reorg_{}'.format(idx), Reorg(stride))
    elif mtype =='linear':
        if 'batch_normalize' in mdef:
            bn = int(mdef['batch_normalize'])  
        else:
            bn = 0
            mdef['batch_normalize'] = 0        
        if 'in_filters' in mdef:
            in_filters = int(mdef['in_filters'])
        else:
            in_filters = o_filters[-1]
        filters = int(mdef['filters'])
        linear = nn.Linear(in_filters, filters)
        modules.add_module('linear_{}'.format(idx), linear)
        if bn:
            modules.add_module('batch_norm_{}'.format(idx), 
                               nn.BatchNorm1d(filters))
        mact = mdef['activation']
        if mact == 'leaky':
            modules.add_module('leaky_{}'.format(idx), 
                               nn.LeakyReLU(0.1, inplace=True))
        elif mact == 'relu':
           modules.add_module('relu_{}'.format(idx), 
                              nn.ReLU(inplace=True)) 
        elif mact == 'linear':
               pass
        else:
            raise ValueError('Unknown activation type \'{}\''.format(mact))
    else: 
        raise ValueError('Unknown module type \'{}\''.format(mtype))
    return modules, filters

# ========================================================
# MODULES
# ========================================================

class Reorg(nn.Module):
    def __init__(self, stride):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert(x.dim() == 4)
        B, C, H, W = x.size()
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class EmptyModule(nn.Module):
    """ Placeholder layer """
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x