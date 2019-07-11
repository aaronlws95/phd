import torch
import torch.nn     as nn

from src.components import Network

class ZNB_Lift_net(nn.Module):
    def __init__(self, pp_conv_net_cfgfile, pp_lin_net_cfgfile,
                 vp_conv_net_cfgfile, vp_lin_net_cfgfile):
        super().__init__()
        self.avgpool                = nn.AvgPool2d(kernel_size=8, 
                                                   stride=8, 
                                                   padding=1)
        self.pose_prior_conv_net    = Network(pp_conv_net_cfgfile)
        self.view_point_conv_net    = Network(vp_conv_net_cfgfile)
        self.pose_prior_lin_net     = Network(pp_lin_net_cfgfile)
        self.view_point_lin_net     = Network(vp_lin_net_cfgfile)
        
    def forward(self, x, hand_side):
        x               = self.avgpool(x)
        hand_side       = hand_side.unsqueeze(1)
        pred_xyz_canon  = self.pose_prior_conv_net(x)[0]
        pred_xyz_canon  = pred_xyz_canon.view(x.shape[0], -1)
        pred_xyz_canon  = torch.cat((pred_xyz_canon, hand_side), dim=1)
        pred_xyz_canon  = self.pose_prior_lin_net(pred_xyz_canon)[0]
        pred_xyz_canon  = pred_xyz_canon.view(x.shape[0], 21, 3)
        
        pred_viewpoint  = self.view_point_conv_net(x)[0]
        pred_viewpoint  = pred_viewpoint.view(x.shape[0], -1)
        pred_viewpoint  = torch.cat((pred_viewpoint, hand_side), dim=1)
        ux, uy, uz      = self.view_point_lin_net(pred_viewpoint)
        return pred_xyz_canon, ux, uy, uz