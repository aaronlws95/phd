import os
import torch.nn as nn
import torch
import numpy as np
import sys

from .BaseModel import BaseModel
from .networks.darknet import Darknet
from .loss.HPOLoss import HPOLoss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import HPO_utils as HPO
from utils import FPHA_utils as FPHA
from utils.eval_utils import *
from utils.image_utils import *

class HPOModel(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, determinsitic, logger=None):
        super(HPOModel, self).__init__(conf, device, train_mode, exp_dir, determinsitic, logger)
        net = self.get_net(load_epoch, conf["cfg_file"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.loss                   = HPOLoss(conf, device)
        self.seen                   = self.load_seen(load_epoch)
        self.hand_root              = conf["hand_root"]
        
        # validation
        self.val_xyz_L2_error       = 0
        self.val_xyz_21_error       = []
        self.val_loss               = 0
        self.val_len                = 0
        
        # prediction
        if not train_mode:
            self.best_pred_uvd_list = []
            self.topk_pred_uvd_list = []
            self.pred_conf_list     = []
            self.load_epoch         = load_epoch
            
    def get_net(self, load_epoch, cfgfile, pretrain):
        net = Darknet(cfgfile)
        if load_epoch == 0:
            if pretrain[-7:] == "weights":
                net.load_weights(pretrain)
                if self.logger:
                    self.logger.log("LOADED yolov2 WEIGHTS")
            elif pretrain[-5:] == "state":
                ckpt = self.get_ckpt(pretrain)
                self.load_ckpt(net, ckpt["model_state_dict"])
                if self.logger:
                    self.logger.log(f"LOADED {pretrain} WEIGHTS")                
        return net

    def load_seen(self, load_epoch):
        if load_epoch != 0 :
            load_dir = os.path.join(self.save_ckpt_dir, f"model_{load_epoch}.state")
            ckpt = self.get_ckpt(load_dir)
            seen =  ckpt["seen"]
        else:
            seen = 0
        return seen
    
    def predict(self, data_load):
        img = data_load[0]
        return self.net(img)
   
    def get_loss(self, data_load, out, train_out):
        _, uvd_gt = data_load
        return self.loss(out, uvd_gt, train_out)

    def train_step(self, data_load):
        out = self.predict(data_load)
        loss, loss_u, loss_v, loss_d, loss_conf = self.get_loss(data_load, out, True)
        self.seen += out.shape[0]
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {
            "loss": loss.item(),
            "loss_u": loss_u.item(),
            "loss_v": loss_v.item(),
            "loss_d": loss_d.item(),
            "loss_conf": loss_conf.item()
        }
        return loss_dict
    
    def valid_step(self, data_load):
        img = data_load[0]
        output = self.net(img)
        self.val_loss += self.get_loss(data_load, output, False).item()
        self.val_len += 1
        W = output.shape[2]
        H = output.shape[3]
        D = 5
        grid_size = W*H*D
        
        output = output.reshape(64, -1)
        pred_uvd = output[:63, :].reshape(21, 3, grid_size)
        pred_conf = output[63, :].reshape(grid_size)
        pred_conf = torch.sigmoid(pred_conf)
        
        index = torch.from_numpy(np.asarray(np.unravel_index(np.arange(grid_size), (W, H, D)))).type(torch.FloatTensor)
        u = index[0, :].unsqueeze(0).expand(21, -1)
        v = index[1, :].unsqueeze(0).expand(21, -1)
        z = index[2, :].unsqueeze(0).expand(21, -1)       
        
        if self.device != "cpu":
            u = u.cuda()
            v = v.cuda()
            z = z.cuda()        
        
        pred_uvd[self.hand_root, :, :] = torch.sigmoid(pred_uvd[self.hand_root, :, :])
        pred_uvd[:, 0, :] = (pred_uvd[:, 0, :] + u) / W
        pred_uvd[:, 1, :] = (pred_uvd[:, 1, :] + v) / H
        pred_uvd[:, 2, :] = (pred_uvd[:, 2, :] + z) / D
        
        top_idx = torch.topk(pred_conf, 1)[1]
        best_pred_uvd = pred_uvd[:, :, top_idx].squeeze().cpu().numpy()
        
        uvd_gt = data_load[1].cpu().numpy()
        uvd_gt = scale_points_WH(uvd_gt, (1, 1), (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT))
        uvd_gt[..., 2] *= FPHA.REF_DEPTH
        best_pred_uvd = scale_points_WH(best_pred_uvd, (1, 1), (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT))
        best_pred_uvd[..., 2] *= FPHA.REF_DEPTH
        best_pred_xyz = FPHA.uvd2xyz_color(best_pred_uvd)
        xyz_gt = FPHA.uvd2xyz_color(uvd_gt)

        self.val_xyz_L2_error += mean_L2_error(best_pred_xyz, xyz_gt)
        self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(best_pred_xyz-xyz_gt), axis=-1) + 1e-8 ))
        
    def get_valid_loss(self):
        val_xyz_l2_error = self.val_xyz_L2_error / self.val_len
        val_loss = self.val_loss / self.val_len

        pck = get_pck(np.squeeze(np.asarray(self.val_xyz_21_error)))
        
        thresholds = np.arange(0, 85, 5)
        auc = calc_auc(pck, thresholds)

        val_loss_dict = {
            "val_loss": val_loss,
            "xyz_l2_error": val_xyz_l2_error,
            "AUC_0_85": auc 
        }        
        
        if self.logger:
            self.logger.log("VALIDATION LOSS")
            for name, loss in val_loss_dict.items():
                self.logger.log("{}: {}".format(name.upper(), loss))
        
        self.val_loss = 0
        self.val_len = 0
        self.val_xyz_L2_error = 0
        self.val_xyz_21_error = []
        return val_loss_dict
        
    def predict_step(self, data_load):
        img = data_load[0]
        output = self.net(img)

        W = output.shape[2]
        H = output.shape[3]
        D = 5
        self.grid_size = W*H*D
        
        output = output.reshape(64, -1)
        pred_uvd = output[:63, :].reshape(21, 3, self.grid_size)
        pred_conf = output[63, :].reshape(self.grid_size)
        pred_conf = torch.sigmoid(pred_conf)
        
        index = torch.from_numpy(np.asarray(np.unravel_index(np.arange(self.grid_size), (W, H, D)))).type(torch.FloatTensor)
        u = index[0, :].unsqueeze(0).expand(21, -1)
        v = index[1, :].unsqueeze(0).expand(21, -1)
        z = index[2, :].unsqueeze(0).expand(21, -1)       
        
        if self.device != "cpu":
            u = u.cuda()
            v = v.cuda()
            z = z.cuda()        
        
        pred_uvd[self.hand_root, :, :] = torch.sigmoid(pred_uvd[self.hand_root, :, :])
        pred_uvd[:, 0, :] = (pred_uvd[:, 0, :] + u) / W
        pred_uvd[:, 1, :] = (pred_uvd[:, 1, :] + v) / H
        pred_uvd[:, 2, :] = (pred_uvd[:, 2, :] + z) / D
        
        topk_pred_uvd = []
        best_pred_uvd = []
        topk_idx = torch.topk(pred_conf, 10)[1]
        for idx in topk_idx:
            topk_pred_uvd.append(pred_uvd[:, :, idx].cpu().numpy())
        self.best_pred_uvd_list.append(topk_pred_uvd[0])
        self.topk_pred_uvd_list.append(topk_pred_uvd)
        self.pred_conf_list.append(pred_conf.cpu().numpy())
        
    def save_predictions(self, data_split):
        write_dir = os.path.join(self.save_dir, "predict_{}_{}_best.txt".format(self.load_epoch, data_split))
        if self.logger.log:
            self.logger.log(f"WRITING PREDICTIONS {write_dir}")        
        HPO.write_pred(write_dir, self.best_pred_uvd_list, (-1, 63))
        
        write_dir = os.path.join(self.save_dir, "predict_{}_{}_topk.txt".format(self.load_epoch, data_split))
        if self.logger.log:
            self.logger.log(f"WRITING PREDICTIONS {write_dir}")        
        HPO.write_pred(write_dir, self.topk_pred_uvd_list, (-1, 630))
        
        write_dir = os.path.join(self.save_dir, "predict_{}_{}_conf.txt".format(self.load_epoch, data_split))
        if self.logger.log:
            self.logger.log(f"WRITING PREDICTIONS {write_dir}")        
        HPO.write_pred(write_dir, self.pred_conf_list, (-1, self.grid_size))
        if self.logger.log:
            self.logger.log("FINISHED WRITING PREDICTIONS")    
            
        self.best_pred_uvd_list = []
        self.topk_pred_uvd_list = []
        self.pred_conf_list = []                

    def save_ckpt(self, epoch):
        if self.logger:
            self.logger.log(f"SAVING CHECKPOINT EPOCH {epoch}")
        if isinstance(self.net, nn.parallel.DistributedDataParallel):
            network = self.net.module
        else:
            network = self.net
        model_state_dict = network.state_dict()
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        state = {"epoch": epoch, 
                 "model_state_dict": model_state_dict, 
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "seen": self.seen}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))