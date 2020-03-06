import torch
from torch              import nn
from pathlib            import Path
from torch.nn.init      import normal_, constant_
 
from src.components     import Network

class CPM_net(nn.Module):
    def __init__(self):
        super().__init__()
        net_cfgpath     = Path(__file__).resolve().parents[2] /'net_cfg'    
        self.model0     = Network(net_cfgpath/'cpm_vgg19_block0.cfg')

        self.model1_2   = Network(net_cfgpath/'cpm_vgg19_block1_2.cfg')
        self.model2_2   = Network(net_cfgpath/'cpm_vgg19_block2_2.cfg')
        self.model3_2   = Network(net_cfgpath/'cpm_vgg19_block2_2.cfg')
        self.model4_2   = Network(net_cfgpath/'cpm_vgg19_block2_2.cfg')
        self.model5_2   = Network(net_cfgpath/'cpm_vgg19_block2_2.cfg')
        self.model6_2   = Network(net_cfgpath/'cpm_vgg19_block2_2.cfg')
        
        self.initialize_weights_norm()

    def forward(self, x):
        saved_for_loss = []
        out1 = self.model0(x)[0]

        out1_2 = self.model1_2(out1)[0]
        out2 = torch.cat([out1_2, out1], 1)
        saved_for_loss.append(out1_2)
        
        out2_2 = self.model2_2(out2)[0]
        out3 = torch.cat([out2_2, out1], 1)
        saved_for_loss.append(out2_2)

        out3_2 = self.model3_2(out3)[0]
        out4 = torch.cat([out3_2, out1], 1)
        saved_for_loss.append(out3_2)

        out4_2 = self.model4_2(out4)[0]
        out5 = torch.cat([out4_2, out1], 1)
        saved_for_loss.append(out4_2)

        out5_2 = self.model5_2(out5)[0]
        out6 = torch.cat([out5_2, out1], 1)
        saved_for_loss.append(out5_2)

        out6_2 = self.model6_2(out6)[0]
        saved_for_loss.append(out6_2)

        return saved_for_loss, (out6_2)

    def initialize_weights_norm(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_(m.weight, std=0.01)
                # mobilenet conv2d doesn't add bias
                if m.bias is not None:  
                    constant_(m.bias, 0.0)
        
        normal_(self.model1_2.module_list[-1][0].weight, std=0.01)
        normal_(self.model2_2.module_list[-1][0].weight, std=0.01)
        normal_(self.model3_2.module_list[-1][0].weight, std=0.01)
        normal_(self.model4_2.module_list[-1][0].weight, std=0.01)
        normal_(self.model5_2.module_list[-1][0].weight, std=0.01)
        normal_(self.model6_2.module_list[-1][0].weight, std=0.01)