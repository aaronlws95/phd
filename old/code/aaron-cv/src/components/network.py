import torch
import torch.nn     as nn

from src.components import get_module
from src.utils      import parse_model_cfg

def create_network(module_defs):
    module_list = nn.ModuleList()
    net_info = module_defs.pop(0)
    out_filters = [int(net_info['channels'])]
    for i, mdef in enumerate(module_defs):
        modules, filters = get_module(mdef, out_filters, i)
        module_list.append(modules)
        out_filters.append(filters)
    return module_list

class Network(nn.Module):
    def __init__(self, cfgfile):
        super(Network, self).__init__()
        self.module_defs = parse_model_cfg(cfgfile)
        self.module_list = create_network(self.module_defs)

    def forward(self, x):
        layer_outputs = []
        output = []
        for mdef, module in zip(self.module_defs, self.module_list):
            mtype = mdef['type']
            print(x.shape)
            if mtype in ['convolutional', 
                         'dropout',
                         'avgpool',
                         'maxpool', 
                         'reorg']:
                x = module(x)
            elif mtype == 'linear':
                x = x.view(x.shape[0], -1)
                x = module(x)
            elif mtype == 'concat':
                layer_i = [int(x) for x in mdef['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'route':
                layer_i = [int(x) for x in mdef['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            else:
                raise ValueError(f"{mtype} is not specified in forward")

            if 'output' in mdef:
                if int(mdef['output']) == 1:
                    output.append(x)
                      
            layer_outputs.append(x)
        return output

    def info(self):
        """Plot line by line description of model info"""
        # Number of parameters
        n_p = sum(x.numel() for x in self.parameters())  
        # Number of gradients
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)  
        pr = ('layer', 'name', 'gradient', 'parameters', 'shape')
        print("\n{:>5} {:>40} {:>9}{:>12} {:>20}".format(*pr))
        for i, (name, p) in enumerate(self.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s' 
                  %(i, name, p.requires_grad, p.numel(), list(p.shape)))
        pr = (i + 1, n_p, n_g)
        print("Model Summary: {} layers, {} parameters, {} gradients"
              .format(*pr))