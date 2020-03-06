import torchvision
from torch              import nn
from torch.nn.init      import normal_, constant_

from src.utils          import TSN

class TSN_EK_net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        new_length          = None
        before_softmax      = True
        
        num_class_verb      = int(cfg['num_class_verb'])
        num_class_noun      = int(cfg['num_class_noun'])
        base_model          = cfg['base_model']
        self.modality       = cfg['modality']
        self.num_segments   = int(cfg['num_segments'])
        self.dropout        = float(cfg['dropout'])
        self.consensus_type = cfg['consensus_type']
        self.base_weight    = cfg['pretrain']
        partial_bn          = cfg['partial_bn']
        self.reshape        = True
        self.before_softmax = before_softmax
        self.crop_num       = 1
        
        if not before_softmax and self.consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if self.modality == "RGB" else 5
        else:
            self.new_length = new_length

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class_verb, num_class_noun)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus_noun = TSN.ConsensusModule(self.consensus_type)
        self.consensus_verb = TSN.ConsensusModule(self.consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class_verb, num_class_noun):
        if self.dropout == 0:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            delattr(self.base_model, self.base_model.last_layer_name)
            self.base_model._op_list = self.base_model._op_list[:-1]
            self.new_fc_verb = nn.Linear(feature_dim, num_class_verb)
            self.new_fc_noun = nn.Linear(feature_dim, num_class_noun)
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc_verb = nn.Linear(feature_dim, num_class_verb)
            self.new_fc_noun = nn.Linear(feature_dim, num_class_noun)

        std = 0.001
        normal_(self.new_fc_verb.weight, 0, std)
        constant_(self.new_fc_verb.bias, 0)
        normal_(self.new_fc_noun.weight, 0, std)
        constant_(self.new_fc_noun.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        if base_model == 'BNInception':
            from src.components import BNInception_net
            self.base_model = BNInception_net(weight_url=self.base_weight)
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std  = [1]
            self.base_model.last_layer_name = 'fc'
            
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_EK_net, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight   = []
        first_conv_bias     = []
        normal_weight       = []
        normal_bias         = []
        bn                  = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        
        if self.dropout == 0:
            base_out = base_out.squeeze()
        
        base_out_verb = self.new_fc_verb(base_out)
        base_out_noun = self.new_fc_noun(base_out)
        
        if not self.before_softmax:
            base_out_verb = self.softmax(base_out_verb)
            base_out_noun = self.softmax(base_out_noun)

        if self.reshape:
            base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
            base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])

        output_verb = self.consensus_verb(base_out_verb)
        output_noun = self.consensus_noun(base_out_noun)
        
        return output_verb.squeeze(1), output_noun.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size*256//224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([TSN.GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   TSN.GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([TSN.GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   TSN.GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([TSN.GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   TSN.GroupRandomHorizontalFlip(is_flow=False)])