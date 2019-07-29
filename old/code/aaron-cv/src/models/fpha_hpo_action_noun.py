import torch
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Model
from src.utils                      import YOLO, IMG, FPHA
from src.loss                       import get_loss
from src.datasets                   import get_dataset, get_dataloader

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val    = 0
        self.avg    = 0
        self.sum    = 0
        self.count  = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum    += val*n
        self.count  += n
        self.avg    = self.sum/self.count

class FPHA_HPO_Action_Noun(Model):
    """ Hand keypoint estimation with HPO method """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)
        
        self.load_weights(self.load_epoch)

        self.loss       = get_loss(cfg['loss'], cfg)
        self.hand_root  = int(cfg['hand_root'])
        self.num_action = int(cfg['num_action'])
        self.num_noun   = int(cfg['num_noun'])
        
        # Training
        if self.training:
            dataset_kwargs  = {'split_set': cfg['train_set']}
            train_dataset   = get_dataset(cfg, dataset_kwargs)
            self.train_sampler   = None
            shuffle         = cfg['shuffle']
            kwargs = {'batch_size'  :   int(cfg['batch_size']),
                      'shuffle'     :   shuffle,
                      'num_workers' :   int(cfg['num_workers']),
                      'pin_memory'  :   True}
            self.train_loader   = get_dataloader(train_dataset,
                                                 self.train_sampler,
                                                 kwargs)
            # Validation
            dataset_kwargs              = {'split_set': cfg['val_set']}
            val_dataset                 = get_dataset(cfg, dataset_kwargs)
            self.val_loader             = get_dataloader(val_dataset,
                                                         None,
                                                         kwargs)
            self.val_xyz_21_error       = []
            self.top1_action            = AverageMeter()
            self.top5_action            = AverageMeter()
            self.top1_noun              = AverageMeter()
            self.top5_noun              = AverageMeter()
        # Prediction
        else:
            self.best_pred_uvd_list     = []
            self.topk_pred_uvd_list     = []
            self.pred_conf_list         = []
            self.top5_pred_action       = []
            self.top5_pred_noun         = []
            self.action_class_dist      = []
            self.noun_class_dist        = []

    # ========================================================
    # LOADING
    # ========================================================

    def load_weights(self, load_epoch):
        if load_epoch == 0:
            if self.pretrain:
                if 'state' in self.pretrain:
                    load_dir = Path(self.data_dir)/self.pretrain
                    ckpt = self.load_ckpt(load_dir)
                    # Load pretrained model with non-exact layers
                    state_dict = ckpt['model_state_dict']
                    cur_dict = self.net.state_dict()
                    # Filter out unnecessary keys
                    pretrained_dict = \
                        {k: v for k, v in state_dict.items() if k in cur_dict 
                         and k != 'module_list.30.conv_30.weight'
                         and k != 'module_list.30.conv_30.bias'}
                    # Overwrite entries in the existing state dict
                    cur_dict.update(pretrained_dict)
                    # Load the new state dict
                    self.net.load_state_dict(cur_dict)
                else:
                    pt_path = str(Path(self.data_dir)/self.pretrain)
                    YOLO.load_darknet_weights(self.net, pt_path)
        else:
            load_dir = Path(self.save_ckpt_dir)/f'model_{load_epoch}.state'
            ckpt = self.load_ckpt(load_dir)
            self.net.load_state_dict(ckpt['model_state_dict'])
            if self.training:
                self.optimizer.load_state_dict(
                    ckpt['optimizer_state_dict'])
                if self.scheduler:
                    self.scheduler.load_state_dict(
                        ckpt['scheduler_state_dict'])

    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, uvd_gt, action_gt, noun_gt     = data_load
        batch_size                          = img.shape[0]
        img                                 = img.cuda()
        uvd_gt                              = uvd_gt.cuda()
        action_gt                           = action_gt.cuda()
        noun_gt                             = noun_gt.cuda()
        out                                 = self.net(img)[0]
        loss, *hand_losses                  = self.loss(out, uvd_gt, 
                                                        action_gt, noun_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_u, loss_v, loss_d, loss_conf, loss_action, loss_noun = hand_losses
        loss_dict = {
            'loss'          : loss.item(),
            'loss_u'        : loss_u.item(),
            'loss_v'        : loss_v.item(),
            'loss_d'        : loss_d.item(),
            'loss_conf'     : loss_conf.item(),
            'loss_action'   : loss_action.item(),
            'loss_noun'     : loss_noun.item(),
        }

        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, uvd_gt, action_gt, noun_gt     = data_load
        uvd_gt                  = uvd_gt.numpy()
        batch_size              = img.shape[0]
        img                     = img.cuda()
        action_gt               = action_gt.cuda()
        noun_gt                 = noun_gt.cuda()
        pred_hand               = self.net(img)[0]
        W                       = pred_hand.shape[3]
        H                       = pred_hand.shape[2]
        D                       = 5
        pred_hand               = pred_hand.view(batch_size, 64 + self.num_action + self.num_noun, D, H, W)
        pred_hand               = pred_hand.permute(0, 1, 3, 4, 2)

        # Hand
        uvd_gt                  = IMG.scale_points_WH(uvd_gt, (1, 1),
                                                      (FPHA.ORI_WIDTH, 
                                                       FPHA.ORI_HEIGHT))
        uvd_gt[..., 2]          *= FPHA.REF_DEPTH
        xyz_gt                  = FPHA.uvd2xyz_color(uvd_gt)

        for batch in range(batch_size):
            cur_pred_hand   = pred_hand[batch]
            pred_uvd        = cur_pred_hand[:63, :, :, :].view(21, 3, H, W, D)
            pred_conf       = torch.sigmoid(cur_pred_hand[63, :, :, :])
            pred_action     = cur_pred_hand[64:(64 + self.num_action), :, :, :]
            pred_noun       = cur_pred_hand[(64 + self.num_action):, :, :, :]
            
            FT              = torch.FloatTensor
            yv, xv, zv      = torch.meshgrid([torch.arange(H),
                                              torch.arange(W),
                                              torch.arange(D)])
            grid_x          = xv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_y          = yv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_z          = zv.repeat((21, 1, 1, 1)).type(FT).cuda()

            pred_uvd[self.hand_root, :, :, :, :] = \
                torch.sigmoid(pred_uvd[self.hand_root, :, :, :, :])
            pred_uvd[:, 0, :, :, :] = (pred_uvd[:, 0, :, :, :] + grid_x)/W
            pred_uvd[:, 1, :, :, :] = (pred_uvd[:, 1, :, :, :] + grid_y)/H
            pred_uvd[:, 2, :, :, :] = (pred_uvd[:, 2, :, :, :] + grid_z)/D

            pred_uvd                = pred_uvd.contiguous().view(21, 3, -1)
            pred_conf               = pred_conf.contiguous().view(-1)
            
            top_idx                 = torch.topk(pred_conf, 1)[1]
            best_pred_uvd           = pred_uvd[:, :, top_idx].squeeze().cpu().numpy()
            best_pred_uvd           = IMG.scale_points_WH(best_pred_uvd, 
                                                          (1, 1),
                                                          (FPHA.ORI_WIDTH, 
                                                           FPHA.ORI_HEIGHT))
            best_pred_uvd[..., 2]   *= FPHA.REF_DEPTH
            best_pred_xyz           = FPHA.uvd2xyz_color(best_pred_uvd)
            cur_xyz_gt              = xyz_gt[batch]

            pred_action             = pred_action.contiguous().view(self.num_action, -1)[:, top_idx]
            pred_noun               = pred_noun.contiguous().view(self.num_noun, -1)[:, top_idx]
            pred_action             = pred_action.squeeze().unsqueeze(0)
            pred_noun               = pred_noun.squeeze().unsqueeze(0)
            cur_action_gt           = action_gt[batch].unsqueeze(0)
            cur_noun_gt             = noun_gt[batch].unsqueeze(0)
            prec1_act, prec5_act    = self.accuracy(pred_action, cur_action_gt, topk=(1,5))
            prec1_noun, prec5_noun  = self.accuracy(pred_noun, cur_noun_gt, topk=(1,5))
            self.top1_action.update(prec1_act.item(), 1)
            self.top5_action.update(prec5_act.item(), 1)
            self.top1_noun.update(prec1_noun.item(), 1)
            self.top5_noun.update(prec5_noun.item(), 1)

            self.val_xyz_21_error.append(np.sqrt(np.sum(np.square(
                best_pred_xyz-cur_xyz_gt), axis=-1) + 1e-8 ))

    def get_valid_loss(self):
        eps                 = 1e-5
        val_xyz_l2_error    = np.mean(self.val_xyz_21_error)
        val_xyz_squeezed    = np.squeeze(np.asarray(self.val_xyz_21_error))
        pck                 = FPHA.get_pck(val_xyz_squeezed)
        thresholds          = np.arange(0, 85, 5)
        auc                 = FPHA.calc_auc(pck, thresholds)

        val_loss_dict = {
            'xyz_l2_error'  : val_xyz_l2_error,
            'AUC_0_85'      : auc,
            'Avg Prec_verb@1': self.top1_action.avg,
            'Avg Prec_verb@5': self.top5_action.avg,
            'Avg Prec_noun@1': self.top1_noun.avg,
            'Avg Prec_noun@5': self.top5_noun.avg,                        
        }

        self.val_xyz_21_error = []
        self.top1_action.reset()
        self.top5_action.reset()
        self.top1_noun.reset()
        self.top5_noun.reset()
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img                     = data_load[0]
        img                     = img.cuda()
        pred_hand               = self.net(img)[0]
        batch_size              = img.shape[0]
        W                       = pred_hand.shape[3]
        H                       = pred_hand.shape[2]
        D                       = 5
        pred_hand               = pred_hand.view(batch_size, 64 + self.num_action + self.num_noun, D, H, W)
        pred_hand               = pred_hand.permute(0, 1, 3, 4, 2)

        for batch in range(batch_size):
            # Hand
            cur_pred_hand   = pred_hand[batch]
            pred_uvd        = cur_pred_hand[:63, :, :, :].view(21, 3, H, W, D)
            pred_conf       = torch.sigmoid(cur_pred_hand[63, :, :, :])

            FT              = torch.FloatTensor
            yv, xv, zv      = torch.meshgrid([torch.arange(H),
                                              torch.arange(W),
                                              torch.arange(D)])
            grid_x          = xv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_y          = yv.repeat((21, 1, 1, 1)).type(FT).cuda()
            grid_z          = zv.repeat((21, 1, 1, 1)).type(FT).cuda()

            pred_uvd[self.hand_root, :, :, :, :] = \
                torch.sigmoid(pred_uvd[self.hand_root, :, :, :, :])
            pred_uvd[:, 0, :, :, :] = (pred_uvd[:, 0, :, :, :] + grid_x)/W
            pred_uvd[:, 1, :, :, :] = (pred_uvd[:, 1, :, :, :] + grid_y)/H
            pred_uvd[:, 2, :, :, :] = (pred_uvd[:, 2, :, :, :] + grid_z)/D

            pred_uvd    = pred_uvd.contiguous().view(21, 3, -1)
            pred_conf   = pred_conf.contiguous().view(-1)

            topk_pred_uvd = []
            best_pred_uvd = []
            topk_idx = torch.topk(pred_conf, 10)[1]
            for idx in topk_idx:
                topk_pred_uvd.append(pred_uvd[:, :, idx].cpu().numpy())
            
            topk_idx                = torch.topk(pred_conf, 1)[1]
            pred_action             = cur_pred_hand[64:(64 + self.num_action), :, :, :]
            pred_noun               = cur_pred_hand[(64 + self.num_action):, :, :, :]
            pred_action             = pred_action.contiguous().view(self.num_action, -1)[:, topk_idx].squeeze()
            pred_noun               = pred_noun.contiguous().view(self.num_noun, -1)[:, topk_idx].squeeze()

            top5_action             = torch.topk(pred_action, 5)[1].cpu().numpy()
            top5_noun               = torch.topk(pred_noun, 5)[1].cpu().numpy()

            self.action_class_dist.append(pred_action.cpu().numpy())
            self.noun_class_dist.append(pred_noun.cpu().numpy())
            self.best_pred_uvd_list.append(topk_pred_uvd[0])
            self.topk_pred_uvd_list.append(topk_pred_uvd)
            self.pred_conf_list.append(pred_conf.cpu().numpy())
            self.top5_pred_action.append(top5_action)
            self.top5_pred_noun.append(top5_noun)
            
    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_best.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.best_pred_uvd_list, (-1, 63)))
        
        pred_save = "predict_{}_{}_topk.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.topk_pred_uvd_list, (-1, 630)))
        
        pred_save = "predict_{}_{}_conf.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.pred_conf_list)

        pred_save = "predict_{}_{}_top5_action.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.top5_pred_action)

        pred_save = "predict_{}_{}_top5_noun.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, self.top5_pred_noun)

        pred_save = "predict_{}_{}_action_dist.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.action_class_dist, (-1, self.num_action)))

        pred_save = "predict_{}_{}_noun_dist.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.noun_class_dist, (-1, self.num_noun)))

        self.action_class_dist      = []
        self.noun_class_dist        = []
        self.best_pred_uvd_list     = []
        self.topk_pred_uvd_list     = []
        self.pred_conf_list         = []
        self.top5_pred_action       = []
        self.top5_pred_noun         = []

    # ========================================================
    # EVAL
    # ========================================================

    def accuracy(self, output, target, topk=(1,)):
        """ Computes the precision@k for the specified values of k """
        maxk        = max(topk)
        batch_size  = target.size(0)
        _, pred     = output.topk(maxk, 1, True, True)
        pred        = pred.t()
        correct     = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0/batch_size))
        return res