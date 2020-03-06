import torch
import yaml
from torch                      import nn
from pathlib                    import Path

from src.components             import get_basic_layer, parse_expr
from src.utils import DATA_DIR

class BNInception_net(nn.Module):
    def __init__(self,
                 model_path='net_cfg/bn_inception.yaml',
                 weight_url='bn_inception_update.pth'):
        super().__init__()
        cur_path    = Path(__file__).resolve().parents[2]
        model_path  = cur_path/model_path
        weight_url  = Path(DATA_DIR)/weight_url
        manifest    = yaml.load(open(model_path))
        layers      = manifest['layers']
        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat':
                id, out_name, module, out_channel, in_name = \
                    get_basic_layer(l, 3 if len(self._channel_dict) == 0 
                                    else self._channel_dict[in_var[0]], 
                                    conv_bias=True)
                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel

        weights = torch.load(weight_url)
        self.load_state_dict(weights)

    def forward(self, input):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        def get_hook(name):
            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())
            return hook
        
        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = \
                        torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x, data_dict[x].size())
                    raise
    
        return data_dict[self._op_list[-1][2]]