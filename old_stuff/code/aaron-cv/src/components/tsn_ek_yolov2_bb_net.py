import torchvision
from torch              import nn
from torch.nn.init      import normal_, constant_
from pathlib            import Path

from src.utils          import TSN, DATA_DIR, YOLO
from src.components     import Network

class TSN_EK_YOLOV2_BB_net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        new_length          = None
        before_softmax      = True
        
        num_class_verb      = int(cfg['num_class_verb'])
        num_class_noun      = int(cfg['num_class_noun'])
        self.modality       = cfg['modality']
        self.num_segments   = int(cfg['num_segments'])
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

        cur_path            = Path(__file__).resolve().parents[2] 
        net_cfgfile         = cur_path/'net_cfg'/(cfg['net_cfg'] + '.cfg')
        self.base_model     = Network(net_cfgfile)
        self.input_size = int(cfg['input_size'])
        self.input_mean = [1]
        self.input_std  = [1]
        if self.base_weight is not None:
            pt_path = str(Path(DATA_DIR)/self.base_weight)
            YOLO.load_darknet_weights(self.base_model, pt_path)
        
        conv_list = [m for m in list(self.base_model.modules()) if isinstance(m, nn.Conv2d)]
        feature_dim = conv_list[-1].out_channels
        self.new_fc_verb = nn.Linear(feature_dim, num_class_verb)
        self.new_fc_noun = nn.Linear(feature_dim, num_class_noun)

        std = 0.001
        normal_(self.new_fc_verb.weight, 0, std)
        constant_(self.new_fc_verb.bias, 0)
        normal_(self.new_fc_noun.weight, 0, std)
        constant_(self.new_fc_noun.bias, 0)

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

        # output spatial dim will be (input_size/32)/kernel_size
        # e.g (224/32)/7 = 1 
        self.avgpool = pool = nn.AvgPool2d(self.input_size//32, 1, 0, ceil_mode=True)
        
        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN_EK_YOLOV2_BB_net, self).train(mode)
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

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))[0]
        base_out = self.avgpool(base_out)
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