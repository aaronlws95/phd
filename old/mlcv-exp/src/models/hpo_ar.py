import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src import ROOT
from src.models.base_model import Base_Model
from src.datasets import get_dataloader, get_dataset
from src.networks.hpo_ar_net import HPO_AR_Net
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.loss.hpo_ar_loss import HPO_AR_Loss
from src.utils import *

class HPO_AR(Base_Model):
    def __init__(self, cfg, mode, load_epoch):
        super().__init__(cfg, mode, load_epoch)
        self.net        = HPO_AR_Net(cfg).cuda()
        self.optimizer  = get_optimizer(cfg, self.net)
        self.scheduler  = get_scheduler(cfg, self.optimizer)

        self.train_dataloader   = get_dataloader(cfg, get_dataset(cfg, 'train'))
        self.val_dataloader     = get_dataloader(cfg, get_dataset(cfg, 'val'))

        self.pretrain = cfg['pretrain']
        self.load_weights()

        self.loss       = HPO_AR_Loss(cfg)
        self.consensus  = cfg['consensus']
        self.num_action = int(cfg['num_actions'])
        self.num_obj    = int(cfg['num_objects'])

        self.val_top1_action    = Average_Meter()
        self.val_action         = Average_Meter()
        self.val_top1_obj       = Average_Meter()
        self.val_obj            = Average_Meter()

        self.pred_action        = []
        self.pred_obj           = []
        self.action_class_dist  = []
        self.obj_class_dist     = []

        self.dataset = cfg['dataset']

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, action_gt, obj_gt  = data_load
        img                     = img.cuda()
        action_gt               = action_gt.cuda()
        obj_gt                  = obj_gt.cuda()
        out                     = self.net(img)
        loss, *other_losses     = self.loss(out, action_gt, obj_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_action, loss_obj = other_losses
        loss_dict = {
            'loss'          : '{:04f}'.format(loss.item()),
            'loss_action'   : '{:04f}'.format(loss_action.item()),
            'loss_obj'      : '{:04f}'.format(loss_obj.item()),
        }

        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, action_gt, obj_gt = data_load

        bs          = img.shape[0]
        img         = img.cuda()
        action_gt   = action_gt.cuda()
        obj_gt      = obj_gt.cuda()
        pred        = self.net(img)
        W           = pred.shape[3]
        H           = pred.shape[2]
        D           = 5
        pred        = pred.view(bs, self.num_action + self.num_obj, D, H, W)
        pred        = pred.permute(0, 1, 3, 4, 2)

        for batch in range(bs):
            cur_pred        = pred[batch]
            cur_obj_gt      = obj_gt[batch].unsqueeze(0)
            cur_action_gt   = action_gt[batch].unsqueeze(0)

            pred_action = cur_pred[:self.num_action, :, :, :]
            pred_action = pred_action.contiguous().view(self.num_action, -1)
            pred_obj    = cur_pred[self.num_action:, :, :, :]
            pred_obj    = pred_obj.contiguous().view(self.num_obj, -1)

            # consensus
            if self.consensus == 'avg':
                pred_obj    = torch.mean(pred_obj, dim=-1)
                pred_action = torch.mean(pred_action, dim=-1)
            elif self.consensus == 'max':
                pred_obj    = torch.max(pred_obj, dim=-1)
                pred_action = torch.max(pred_action, dim=-1)

            pred_action = pred_action.unsqueeze(0)
            pred_obj    = pred_obj.unsqueeze(0)
            prec1_act, prec5_act    = topk_accuracy(pred_action, cur_action_gt, topk=(1,5))
            prec1_obj, prec5_obj    = topk_accuracy(pred_obj, cur_obj_gt, topk=(1,5))

            self.val_top1_action.update(prec1_act.item(), 1)
            self.val_action.update(prec5_act.item(), 1)
            self.val_top1_obj.update(prec1_obj.item(), 1)
            self.val_obj.update(prec5_obj.item(), 1)

    def get_valid_loss(self):
        val_loss_dict = {
            'top1_act'  : self.val_top1_action.avg,
            'top5_act'  : self.val_action.avg,
            'top1_obj'  : self.val_top1_obj.avg,
            'top5_obj'  : self.val_obj.avg,
        }

        self.val_top1_action.reset()
        self.val_action.reset()
        self.val_top1_obj.reset()
        self.val_obj.reset()
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img         = data_load[0]
        img         = img.cuda()
        pred        = self.net(img)
        bs          = img.shape[0]
        W           = pred.shape[3]
        H           = pred.shape[2]
        D           = 5
        pred        = pred.view(bs, self.num_action + self.num_obj, D, H, W)
        pred        = pred.permute(0, 1, 3, 4, 2)

        for batch in range(bs):
            cur_pred   = pred[batch]
            pred_action = cur_pred[:self.num_action, :, :, :]
            pred_action = pred_action.contiguous().view(self.num_action, -1)
            pred_obj    = cur_pred[self.num_action:, :, :, :]
            pred_obj    = pred_obj.contiguous().view(self.num_obj, -1)

            # consensus
            if self.consensus == 'avg':
                pred_obj    = torch.mean(pred_obj, dim=-1)
                pred_action = torch.mean(pred_action, dim=-1)
            elif self.consensus == 'max':
                pred_obj    = torch.max(pred_obj, dim=-1)
                pred_action = torch.max(pred_action, dim=-1)

            top_action = torch.topk(pred_action, 1)[1].cpu().numpy()
            top_obj    = torch.topk(pred_obj, 1)[1].cpu().numpy()

            self.action_class_dist.append(pred_action.cpu().numpy())
            self.obj_class_dist.append(pred_obj.cpu().numpy())
            self.pred_action.append(top_action)
            self.pred_obj.append(top_obj)

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_action.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_action)

        pred_save = "predict_{}_{}_obj.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_obj)

        pred_save = "predict_{}_{}_action_dist.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.action_class_dist, (-1, self.num_action)))

        pred_save = "predict_{}_{}_obj_dist.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(ROOT)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.obj_class_dist, (-1, self.num_obj)))

        self.action_class_dist  = []
        self.obj_class_dist     = []
        self.pred_action        = []
        self.pred_obj           = []

    # ========================================================
    # DETECT
    # ========================================================

    def detect(self, img):
        pred        = self.net(img)

        W           = pred.shape[3]
        H           = pred.shape[2]
        D           = 5
        pred        = pred.view(-1, self.num_action + self.num_obj, D, H, W)
        pred        = pred.permute(0, 1, 3, 4, 2)[0]

        pred_action = pred[:self.num_action, :, :, :]
        pred_action = pred_action.contiguous().view(self.num_action, -1)
        pred_obj    = pred[self.num_action:, :, :, :]
        pred_obj    = pred_obj.contiguous().view(self.num_obj, -1)

        # consensus
        if self.consensus == 'avg':
            pred_obj    = torch.mean(pred_obj, dim=-1)
            pred_action = torch.mean(pred_action, dim=-1)
        elif self.consensus == 'max':
            pred_obj    = torch.max(pred_obj, dim=-1)
            pred_action = torch.max(pred_action, dim=-1)

        top_action = torch.topk(pred_action, 1)[1].cpu().numpy()[0]
        top_obj    = torch.topk(pred_obj, 1)[1].cpu().numpy()[0]

        if 'fpha' in self.dataset:
            action_dict    = FPHA.get_action_dict()
            obj_dict       = FPHA.get_obj_dict()
        elif 'ek' in self.dataset:
            action_dict    = EK.get_verb_dict()
            obj_dict       = EK.get_noun_dict()

        print(action_dict[top_action], obj_dict[top_obj])

    def detect_video(self, seq_path, img_size):
        with torch.no_grad():
            seq = [x for x in sorted(seq_path.glob('*')) if x.is_file()]

            frames              = []
            pred_obj_list       = []
            pred_action_list    = []
            for f in tqdm(seq):
                img = get_img_dataloader(str(f), img_size)
                img = img.unsqueeze(0).cuda()
                pred        = self.net(img)

                W           = pred.shape[3]
                H           = pred.shape[2]
                D           = 5
                pred        = pred.view(-1, self.num_action + self.num_obj, D, H, W)
                pred        = pred.permute(0, 1, 3, 4, 2)[0]

                pred_action = pred[:self.num_action, :, :, :]
                pred_action = pred_action.contiguous().view(self.num_action, -1)
                pred_obj    = pred[self.num_action:, :, :, :]
                pred_obj    = pred_obj.contiguous().view(self.num_obj, -1)

                # consensus
                if self.consensus == 'avg':
                    pred_obj_list.append(torch.mean(pred_obj, dim=-1).cpu().numpy())
                    pred_action_list.append(torch.mean(pred_action, dim=-1).cpu().numpy())
                elif self.consensus == 'max':
                    pred_obj_list.append(torch.max(pred_obj, dim=-1).cpu().numpy())
                    pred_action_list.append(torch.max(pred_action, dim=-1).cpu().numpy())
                    
            pred_action = np.mean(pred_action_list, axis=0)
            pred_obj = np.mean(pred_obj_list, axis=0)
            top_action = np.argmax(pred_action)
            top_obj    = np.argmax(pred_obj)

            if 'fpha' in self.dataset:
                action_dict    = FPHA.get_action_dict()
                obj_dict       = FPHA.get_obj_dict()
            elif 'ek' in self.dataset:
                action_dict    = EK.get_verb_dict()
                obj_dict       = EK.get_noun_dict()

            print(action_dict[top_action], obj_dict[top_obj])