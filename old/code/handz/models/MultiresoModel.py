import os
import torch.nn as nn
import torch
import numpy as np
import sys

from .BaseModel import BaseModel
from .networks.multiresonet import MultiresoNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Multireso_utils as MRESO
from utils import FPHA_utils as FPHA
from utils.eval_utils import *
from utils.image_utils import *

class MultiresoModel(BaseModel):
    def __init__(self, conf, device, load_epoch, train_mode, exp_dir, determinsitic, logger=None):
        super(MultiresoModel, self).__init__(conf, device, train_mode, exp_dir, determinsitic, logger)
        net = self.get_net(load_epoch, conf["num_filter"], conf["pretrain"])
        self.init_network(net, load_epoch)
        
        self.loss                   = nn.MSELoss()
        
        # validation
        self.val_xyz_L2_error       = 0
        self.val_xyz_21_error       = []
        self.val_loss               = 0
        self.val_len                = 0
        
        # prediction
        if not train_mode:
            self.pred_uvd_list      = []
            
    def get_net(self, load_epoch, num_filter, pretrain):
        net = MultiresoNet(num_filter)
        if load_epoch == 0:
            if pretrain != "none":
                ckpt = self.get_ckpt(pretrain)
                self.load_ckpt(net, ckpt["model_state_dict"])
                if self.logger:
                    self.logger.log(f"LOADED {pretrain} WEIGHTS")                
        return net

    def predict(self, data_load):
        imgs = data_load
        return self.net(*imgs)
   
    def get_loss(self, data_load, out):
        uvd_gt = data_load[1]
        return self.loss(out, uvd_gt)

    def train_step(self, data_load):
        out = self.predict(data_load)
        loss = self.get_loss(data_load, out)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {
            "loss": loss.item(),
        }
        return loss_dict
    
    def valid_step(self, data_load):
        imgs = data_load[0]
        uvd_gt = data_load[1]
        output = self.net(*img)
        batch_size = uvd_gt.shape[0]
        
        self.val_loss += self.get_loss(data_load, output).item()
        self.val_len += batch_size        
        
        output = output.reshape(-1, 21, 3).cpu().numpy()
        pred_xyz, _ = MRESO.normuvd2xyzuvd_color(output)
        xyz_gt, _ = MRESO.normuvd2xyzuvd_color(uvd_gt)
        
        for i in range(batch_size):
            cur_pred_xyz = pred_xyz[i]
            cur_xyz_gt = xyz_gt[i]

            self.val_xyz_L2_error += mean_L2_error(cur_pred_xyz, cur_xyz_gt)
            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(cur_pred_xyz-cur_xyz_gt), axis=-1) + 1e-8 ))
        
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
        imgs = data_load[0]
        output = self.net(*img)
        batch_size = uvd_gt.shape[0]
        
        output = output.reshape(-1, 21, 3).cpu().numpy()
        _, pred_uvd = MRESO.normuvd2xyzuvd_color(output)
        
        for i in range(batch_size):
            self.pred_uvd_list.append(pred_uvd[i])
        
    def save_predictions(self, data_split):
        if self.logger:
            self.logger.log("WRITING PREDICTIONS")
        pred_file = os.path.join(self.save_dir, "predict_{}_{}.txt".format(self.load_epoch, data_split))
        np.savetxt(pred_file, self.pred_list) 
        
        self.pred_uvd_list = []             
        if self.logger:
            self.logger.log("FINISHING WRITING PREDICTIONS")
        
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