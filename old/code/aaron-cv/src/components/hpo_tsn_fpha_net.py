import torch
import torchvision
import torch.nn                                     as nn
from torch.nn.init      import normal_, constant_
from pathlib            import Path

from src.components     import Network
from src.utils          import TSN, DATA_DIR

class HPO_TSN_FPHA_net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        num_class           = int(cfg['num_class'])
        self.modality       = cfg['modality']
        self.num_segments   = int(cfg['num_segments'])
        self.consensus_type = cfg['consensus_type']
        self.base_weight    = cfg['pretrain']
        partial_bn          = cfg['partial_bn']
        self.crop_num       = 1
        self.hand_root      = int(cfg['hand_root'])
        self.freeze_base    = cfg['freeze_base']
        cur_path            = Path(__file__).resolve().parents[2] 
        net_cfgfile         = cur_path/'net_cfg'/(cfg['net_cfg'] + '.cfg')
        self.base_model     = Network(net_cfgfile)
        
        save_ckpt_dir       = Path(DATA_DIR)/cfg['base_exp_dir']/'ckpt'
        base_load_epoch     = cfg['base_load_epoch']
        base_load_dir       = Path(save_ckpt_dir)/f'model_{base_load_epoch}.state'
        map_loc             = "cuda:" + cfg['device']
        ckpt                = torch.load(base_load_dir, map_location=map_loc)        
        
        self.base_model.load_state_dict(ckpt['model_state_dict'])
        # Freeze base model
        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        self.input_size     = int(cfg['input_size'])
        self.input_mean     = [1]
        self.input_std      = [1]
        conv_list           = [m for m in list(self.base_model.modules()) \
                               if isinstance(m, nn.Conv2d)]
        feature_dim         = 63

        self.new_fc = nn.Linear(feature_dim, num_class)
        self.consensus = TSN.ConsensusModule(self.consensus_type)

    def forward(self, input):
        sample_len = 3
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))[0]

        bs          = base_out.shape[0]
        W           = base_out.shape[2]
        H           = base_out.shape[3]
        D           = 5
        base_out    = base_out.view(bs, 64, D, H, W)
        base_out    = base_out.permute(0, 1, 3, 4, 2)

        pred_uvd    = base_out[:, :63, :, :, :].view(bs, 21, 3, H, W, D)
        pred_conf   = base_out[:, 63, :, :, :]
        pred_conf   = torch.sigmoid(pred_conf)

        pred_uvd[:, self.hand_root, :, :, :, :] = \
            torch.sigmoid(pred_uvd[:, self.hand_root, :, :, :, :])
            
        pred_uvd_out = pred_uvd.clone().detach()
        
        pred_uvd    = pred_uvd.contiguous().view(bs, 21, 3, -1)
        pred_conf   = pred_conf.contiguous().view(bs, -1)

        top_conf, top_idx = torch.topk(pred_conf, 1)
        best_pred_uvd = torch.zeros(bs, 21, 3).cuda()
        for i, idx in enumerate(top_idx):
            best_pred_uvd[i] = pred_uvd[i, :, :, top_idx[i]].squeeze()
        best_pred_uvd = best_pred_uvd.view(bs, -1)
        base_out = self.new_fc(best_pred_uvd)

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)

        return (pred_uvd_out, top_idx), output.squeeze(1)
    
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