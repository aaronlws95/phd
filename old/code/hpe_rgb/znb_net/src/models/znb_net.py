import torch
import torch.nn as nn
import torch.nn.functional as F

class ZNB_Pose_Net(nn.Module):
  def __init__(self, channels=3):
    super(ZNB_Pose_Net, self).__init__()
    self.ZNB_feature_extr = self._ZNB_feature_extr(channels)
    self.get_out_feat = self._get_out_feat()
    self.ZNB_rec_unit_0 = self._ZNB_rec_unit(149)
    self.get_out_rec_0 = self._get_out_rec()
    self.ZNB_rec_unit_1 = self._ZNB_rec_unit(149)
    self.get_out_rec_1 = self._get_out_rec()

  def _conv_relu(self, num_filter, k, channels, padding):
    conv = nn.Conv2d(channels, num_filter, kernel_size=k, padding=padding)
    relu = nn.ReLU()
    return nn.Sequential(conv, relu)

  def _ZNB_feature_extr(self, c):
    modules = []
    modules.append(self._conv_relu(64, 3, c, 1))
    modules.append(self._conv_relu(64, 3, 64, 1))
    modules.append(nn.MaxPool2d(2))
    modules.append(self._conv_relu(128, 3, 64, 1))
    modules.append(self._conv_relu(128, 3, 128, 1))
    modules.append(nn.MaxPool2d(2))
    modules.append(self._conv_relu(256, 3, 128, 1))
    modules.append(self._conv_relu(256, 3, 256, 1))
    modules.append(self._conv_relu(256, 3, 256, 1))
    modules.append(self._conv_relu(256, 3, 256, 1))
    modules.append(nn.MaxPool2d(2))
    modules.append(self._conv_relu(512, 3, 256, 1))
    modules.append(self._conv_relu(512, 3, 512, 1))

    modules.append(self._conv_relu(256, 3, 512, 1))
    modules.append(self._conv_relu(256, 3, 256, 1))
    modules.append(self._conv_relu(256, 3, 256, 1))
    modules.append(self._conv_relu(256, 3, 256, 1))
    modules.append(self._conv_relu(128, 3, 256, 1))
    return nn.Sequential(*modules)

  def _get_out_feat(self):
    c0 = self._conv_relu(512, 1, 128, padding=0)
    c1 = nn.Conv2d(512, 21, kernel_size=1, padding=0)
    return nn.Sequential(c0, c1)

  def _get_out_rec(self):
    c0 = self._conv_relu(128, 1, 128, padding=0)
    c1 = nn.Conv2d(128, 21, kernel_size=1, padding=0)
    return nn.Sequential(c0, c1)

  def _ZNB_rec_unit(self, c):
    modules = []
    modules.append(self._conv_relu(128, 7, c, 3))
    modules.append(self._conv_relu(128, 7, 128, 3))
    modules.append(self._conv_relu(128, 7, 128, 3))
    modules.append(self._conv_relu(128, 7, 128, 3))
    modules.append(self._conv_relu(128, 7, 128, 3))
    return nn.Sequential(*modules)

  def forward(self, x):
    encoding = self.ZNB_feature_extr(x)
    out0 = self.get_out_feat(encoding)
    x = torch.cat((out0, encoding), dim=1)
    x = self.ZNB_rec_unit_0(x)
    out1 = self.get_out_rec_0(x)
    x = torch.cat((out1, encoding), dim=1)
    x = self.ZNB_rec_unit_1(x)
    out2 = self.get_out_rec_1(x)
    out = [out0, out1, out2]
    return out

class CPM(nn.Module):
    # https://github.com/namedBen/Convolutional-Pose-Machines-Pytorch/blob/master/test/cpm_test.py
    def __init__(self, k):
        super(CPM, self).__init__()
        self.k = k
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_stage1 = nn.Conv2d(512, self.k, kernel_size=1)

        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage2 = nn.Conv2d(32 + self.k, 128, kernel_size=11, padding=5)
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage2 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage3 = nn.Conv2d(32 + self.k, 128, kernel_size=11, padding=5)
        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage3 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage4 = nn.Conv2d(32 + self.k, 128, kernel_size=11, padding=5)
        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage4 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage5 = nn.Conv2d(32 + self.k, 128, kernel_size=11, padding=5)
        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage5 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

        self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_stage6 = nn.Conv2d(32 + self.k, 128, kernel_size=11, padding=5)
        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_stage6 = nn.Conv2d(128, self.k, kernel_size=1, padding=0)

    def _stage1(self, image):

        x = self.pool1_stage1(F.relu(self.conv1_stage1(image)))
        x = self.pool2_stage1(F.relu(self.conv2_stage1(x)))
        x = self.pool3_stage1(F.relu(self.conv3_stage1(x)))
        x = F.relu(self.conv4_stage1(x))
        x = F.relu(self.conv5_stage1(x))
        x = F.relu(self.conv6_stage1(x))
        x = self.conv7_stage1(x)

        return x

    def _middle(self, image):

        x = self.pool1_stage2(F.relu(self.conv1_stage2(image)))
        x = self.pool2_stage2(F.relu(self.conv2_stage2(x)))
        x = self.pool3_stage2(F.relu(self.conv3_stage2(x)))

        return x

    def _stage2(self, pool3_stage2_map, conv7_stage1_map):

        x = F.relu(self.conv4_stage2(pool3_stage2_map))
        x = torch.cat([x, conv7_stage1_map], dim=1)
        x = F.relu(self.Mconv1_stage2(x))
        x = F.relu(self.Mconv2_stage2(x))
        x = F.relu(self.Mconv3_stage2(x))
        x = F.relu(self.Mconv4_stage2(x))
        x = self.Mconv5_stage2(x)

        return x

    def _stage3(self, pool3_stage2_map, Mconv5_stage2_map):

        x = F.relu(self.conv1_stage3(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage2_map], dim=1)
        x = F.relu(self.Mconv1_stage3(x))
        x = F.relu(self.Mconv2_stage3(x))
        x = F.relu(self.Mconv3_stage3(x))
        x = F.relu(self.Mconv4_stage3(x))
        x = self.Mconv5_stage3(x)

        return x

    def _stage4(self, pool3_stage2_map, Mconv5_stage3_map):

        x = F.relu(self.conv1_stage4(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage3_map], dim=1)
        x = F.relu(self.Mconv1_stage4(x))
        x = F.relu(self.Mconv2_stage4(x))
        x = F.relu(self.Mconv3_stage4(x))
        x = F.relu(self.Mconv4_stage4(x))
        x = self.Mconv5_stage4(x)

        return x

    def _stage5(self, pool3_stage2_map, Mconv5_stage4_map):

        x = F.relu(self.conv1_stage5(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage4_map], dim=1)
        x = F.relu(self.Mconv1_stage5(x))
        x = F.relu(self.Mconv2_stage5(x))
        x = F.relu(self.Mconv3_stage5(x))
        x = F.relu(self.Mconv4_stage5(x))
        x = self.Mconv5_stage5(x)

        return x

    def _stage6(self, pool3_stage2_map, Mconv5_stage5_map):

        x = F.relu(self.conv1_stage6(pool3_stage2_map))
        x = torch.cat([x, Mconv5_stage5_map], dim=1)
        x = F.relu(self.Mconv1_stage6(x))
        x = F.relu(self.Mconv2_stage6(x))
        x = F.relu(self.Mconv3_stage6(x))
        x = F.relu(self.Mconv4_stage6(x))
        x = self.Mconv5_stage6(x)

        return x

    def forward(self, image):

        conv7_stage1_map = self._stage1(image)

        pool3_stage2_map = self._middle(image)

        Mconv5_stage2_map = self._stage2(pool3_stage2_map, conv7_stage1_map)
        Mconv5_stage3_map = self._stage3(pool3_stage2_map, Mconv5_stage2_map)
        Mconv5_stage4_map = self._stage4(pool3_stage2_map, Mconv5_stage3_map)
        Mconv5_stage5_map = self._stage5(pool3_stage2_map, Mconv5_stage4_map)
        Mconv5_stage6_map = self._stage6(pool3_stage2_map, Mconv5_stage5_map)

        conv7_stage1_map = F.interpolate(conv7_stage1_map, (256, 256), mode='bilinear')
        Mconv5_stage2_map = F.interpolate(Mconv5_stage2_map, (256, 256), mode='bilinear')
        Mconv5_stage3_map = F.interpolate(Mconv5_stage3_map, (256, 256), mode='bilinear')
        Mconv5_stage4_map = F.interpolate(Mconv5_stage4_map, (256, 256), mode='bilinear')
        Mconv5_stage5_map = F.interpolate(Mconv5_stage5_map, (256, 256), mode='bilinear')
        Mconv5_stage6_map = F.interpolate(Mconv5_stage6_map, (256, 256), mode='bilinear')

        return conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map, Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map

class ZNB_Lift_Net(nn.Module):
  def __init__(self):
    super(ZNB_Lift_Net, self).__init__()
    self.pose_prior_net = ZNB_Pose_Prior_Net()
    self.view_point_net = ZNB_Viewpoint_Net()

  def forward(self, x):
    pred_xyz_canon = self.pose_prior_net(x)
    pred_rot_mat = self.view_point_net(x)
    return pred_xyz_canon, pred_rot_mat

class ZNB_Pose_Prior_Net(nn.Module):
  def __init__(self):
    super(ZNB_Pose_Prior_Net, self).__init__()
    self.pool_scoremap = nn.AvgPool2d(kernel_size=8, stride=8, padding=1)
    self.ZNB_detect_rel_norm_coord = self._ZNB_detect_rel_norm_coord()
    self.ZNB_estim_rel_coord = self._ZNB_estim_rel_coord()

  def _conv_relu(self, num_filter, k, channels, padding, stride=1):
    conv = nn.Conv2d(channels, num_filter, kernel_size=k, padding=padding, stride=stride)
    relu = nn.ReLU()
    return nn.Sequential(conv, relu)

  def _fcn_relu_drop(self, in_feat, out_feat, p):
    fcn = nn.Linear(in_feat, out_feat)
    relu = nn.ReLU()
    dropout = nn.Dropout(p)
    return nn.Sequential(fcn, relu, dropout)

  def _ZNB_detect_rel_norm_coord(self, kpt=21):
    modules = []
    modules.append(self._conv_relu(32, 3, kpt, 1))
    modules.append(self._conv_relu(32, 3, 32, 1, 2))
    modules.append(self._conv_relu(64, 3, 32, 1))
    modules.append(self._conv_relu(64, 3, 64, 1, 2))
    modules.append(self._conv_relu(128, 3, 64, 1))
    modules.append(self._conv_relu(128, 3, 128, 1, 2))
    return nn.Sequential(*modules)

  def _ZNB_estim_rel_coord(self):
    modules = []
    modules.append(self._fcn_relu_drop(2048, 512, 0.2))
    modules.append(self._fcn_relu_drop(512, 512, 0.2))
    modules.append(nn.Linear(512, 63))
    return nn.Sequential(*modules)

  def forward(self, x):
    x = self.pool_scoremap(x)
    encoding = self.ZNB_detect_rel_norm_coord(x)
    out = encoding.reshape(encoding.shape[0], -1)
    out = self.ZNB_estim_rel_coord(out)
    return out.reshape(out.shape[0], 21, 3)

class ZNB_Viewpoint_Net(nn.Module):
  def __init__(self, channels=3):
    super(ZNB_Viewpoint_Net, self).__init__()
    self.pool_scoremap = nn.AvgPool2d(kernel_size=8, stride=8, padding=1)
    self.ZNB_conv_down_scoremap = self._ZNB_conv_down_scoremap()
    self.ZNB_estim_viewpoint = self._ZNB_estim_viewpoint()
    self.estim_ux = nn.Linear(128, 1)
    self.estim_uy = nn.Linear(128, 1)
    self.estim_uz = nn.Linear(128, 1)

  def _conv_relu(self, num_filter, k, channels, padding, stride=1):
    conv = nn.Conv2d(channels, num_filter, kernel_size=k, padding=padding, stride=stride)
    relu = nn.ReLU()
    return nn.Sequential(conv, relu)

  def _fcn_relu_drop(self, in_feat, out_feat, p):
    fcn = nn.Linear(in_feat, out_feat)
    relu = nn.ReLU()
    dropout = nn.Dropout(p)
    return nn.Sequential(fcn, relu, dropout)

  def _ZNB_conv_down_scoremap(self, kpt=21):
    modules = []
    modules.append(self._conv_relu(64, 3, kpt, 1))
    modules.append(self._conv_relu(64, 3, 64, 1, 2))
    modules.append(self._conv_relu(128, 3, 64, 1))
    modules.append(self._conv_relu(128, 3, 128, 1, 2))
    modules.append(self._conv_relu(256, 3, 128, 1))
    modules.append(self._conv_relu(256, 3, 256, 1, 2))
    return nn.Sequential(*modules)

  def _ZNB_estim_viewpoint(self):
    modules = []
    modules.append(self._fcn_relu_drop(4096, 256, 0.25))
    modules.append(self._fcn_relu_drop(256, 128, 0.25))
    return nn.Sequential(*modules)

  def _get_rot_mat(self, ux, uy, uz):
    theta = torch.sqrt(ux*ux + uy*uy + uz*uz + 1e-8)
    sin = torch.sin(theta)
    cos = torch.cos(theta)
    mcos = 1 - torch.cos(theta)
    norm_ux = ux/theta
    norm_uy = uy/theta
    norm_uz = uz/theta

    row_1 = torch.cat((cos+norm_ux*norm_ux*mcos,
    norm_ux*norm_uy*mcos-norm_uz*sin,
    norm_ux*norm_uz*mcos+norm_uy*sin), dim=-1)

    row_2 = torch.cat((norm_uy*norm_ux*mcos+norm_uz*sin,
    cos+norm_uy*norm_uy*mcos,
    norm_uy*norm_uz*mcos-norm_ux*sin), dim=-1)

    row_3 = torch.cat((norm_uz*norm_ux*mcos-norm_uy*sin,
    norm_uz*norm_uy*mcos+norm_ux*sin,
    cos*norm_uz*norm_uz*mcos), dim=-1)


    rot_mat = torch.stack((row_1, row_2, row_3), dim=-1)
    return rot_mat

  def forward(self, x):
    x = self.pool_scoremap(x)
    encoding = self.ZNB_conv_down_scoremap(x)
    out = encoding.reshape(encoding.shape[0], -1)
    out = self.ZNB_estim_viewpoint(out)
    ux = self.estim_ux(out)
    uy = self.estim_uy(out)
    uz = self.estim_uz(out)
    rot_mat = self._get_rot_mat(ux, uy, uz)
    return rot_mat

