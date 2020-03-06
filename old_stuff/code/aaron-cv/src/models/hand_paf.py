import torch
import torchvision
import torch.utils.model_zoo        as model_zoo
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm

from src.models                     import Model
from src.utils                      import DATA_DIR, PAF, IMG
from src.datasets                   import get_dataloader, get_dataset
from src.components                 import get_scheduler, get_optimizer, \
                                           CPM_net

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

class Hand_PAF(Model):
    """ Supposed using PAFs but then I realised that PAFs are only necessary
    for multiple instances """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)

        self.net            = CPM_net()
        self.net            = self.net.cuda()

        if self.training:
            # Optimizer and scheduler
            self.optimizer  = get_optimizer(cfg, self.net)
            self.scheduler  = get_scheduler(cfg, self.optimizer)

        self.load_weights(self.load_epoch)

        # Training
        if self.training:
            # Dataset
            dataset_kwargs = {'split_set': cfg['train_set']}
            train_dataset = get_dataset(cfg, dataset_kwargs)
            self.train_sampler = None
            train_kwargs = {'batch_size'  :   int(cfg['batch_size']),
                            'shuffle'     :   cfg['shuffle'],
                            'num_workers' :   int(cfg['num_workers']),
                            'pin_memory'  :   True}
            self.train_loader = get_dataloader(train_dataset,
                                               self.train_sampler,
                                               train_kwargs)

            # Validation
            val_kwargs =   {'batch_size'  :   int(cfg['batch_size']),
                            'shuffle'     :   False,
                            'num_workers' :   int(cfg['num_workers']),
                            'pin_memory'  :   True}
            dataset_kwargs = {'split_set': cfg['val_set']}
            val_dataset = get_dataset(cfg, dataset_kwargs)
            self.val_loader = get_dataloader(val_dataset,
                                             None,
                                             val_kwargs)
            
            self.val_loss  = AverageMeter()
        else:
            self.output_list    = []

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
                        {k: v for k, v in state_dict.items() if k in cur_dict}
                    # Overwrite entries in the existing state dict
                    cur_dict.update(pretrained_dict)
                    # Load the new state dict
                    self.net.load_state_dict(cur_dict)
                else:
                    if self.pretrain == 'vgg19':
                        url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
                        num_weights = 20
                    state_dict = model_zoo.load_url(url)
                    keys = state_dict.keys()
                    weights_load = {}
                    # weight+bias,weight+bias.....(repeat 10 times)
                    for i in range(num_weights):
                        weights_load[list(self.net.state_dict().keys())[i]] = \
                            state_dict[list(keys)[i]]
                    state = self.net.state_dict()
                    state.update(weights_load)
                    self.net.load_state_dict(state)
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

    def loss(self, saved_for_loss, heatmaps, heatmap_vis):
        
        mseloss = nn.MSELoss(reduction='mean')

        total_loss = 0
        for j in range(len(saved_for_loss)):
            pred = saved_for_loss[j]*heatmap_vis
            gt = heatmaps*heatmap_vis
            total_loss += mseloss(pred, gt)

        return total_loss
        
    def train_step(self, data_load):
        img, heatmaps, heatmap_vis, _   = data_load

        img                     = img.cuda()
        heatmaps                = heatmaps.cuda()
        heatmap_vis             = heatmap_vis.cuda()

        saved_for_loss, _       = self.net(img)
        
        loss                    = self.loss(saved_for_loss, heatmaps, heatmap_vis)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict = {
            'loss' : loss.item()
        }
        
        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, heatmaps, heatmap_vis, _   = data_load

        img                     = img.cuda()
        heatmaps                = heatmaps.cuda()
        heatmap_vis             = heatmap_vis.cuda()
        batch_size              = img.shape[0]
        
        saved_for_loss, _       = self.net(img)
        loss                    = self.loss(saved_for_loss, heatmaps, heatmap_vis)

        self.val_loss.update(loss.item(), batch_size)
    
    def get_valid_loss(self):
        val_loss_dict = {
            'val_loss': self.val_loss.avg,
        }
        
        # self.cur_val_loss = self.val_loss.avg
        self.val_loss.reset()
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        img, heatmaps, _, _   = data_load

        img                     = img.cuda()
        _, out                  = self.net(img)
        
        for batch in range(img.shape[0]):
            cur_out = out[batch]
            cur_out = cur_out.cpu().numpy().transpose((1, 2, 0))
            cur_out = IMG.resize_img(cur_out, (img[0].shape[1], img[0].shape[2]))
            joint_list_per_joint_type = PAF.NMS(cur_out, img[0].shape[1]/float(cur_out.shape[0]))
            joint_list = np.array([tuple(peak) + (joint_type,) for joint_type, 
                                   joint_peaks in enumerate(joint_list_per_joint_type) for peak in joint_peaks])
            joints = np.zeros((21, 2))
            for j in joint_list:
                joints[int(j[-1])] = j[:2]
     
            self.output_list.append(joints)

    def save_predictions(self, data_split):
        pred_save = "predict_{}_{}_uvd.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.output_list, (-1, 42)))
        self.output_list = []
