import torch
import torchvision
import torch.nn                     as nn
import numpy                        as np
from pathlib                        import Path
from tqdm                           import tqdm
from torch.utils.data               import DataLoader
from torch.nn.utils                 import clip_grad_norm_
from sklearn.metrics                import confusion_matrix

from src.models                     import Model
from src.utils                      import IMG, TSN
from src.loss                       import get_loss
from src.datasets                   import TSN_Labels, FPHA_Hand
from src.components                 import get_scheduler, get_optimizer, \
                                           HPO_TSN_FPHA_net
                                           
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

class HPO_TSN_FPHA(Model):
    """ Temporal Stream Network with YOLOV2 backbone """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)

        # Inference settings
        if not self.training:
            cfg['num_segments']     = 1
            
        self.net            = HPO_TSN_FPHA_net(cfg)
        self.net            = self.net.cuda()
        is_train_aug        = cfg['train_aug']
        train_aug           = self.net.get_augmentation()
        
        if self.training:
            # Optimizer and scheduler
            self.optimizer  = get_optimizer(cfg, self.net)
            self.scheduler  = get_scheduler(cfg, self.optimizer)
            
        self.loss           = nn.CrossEntropyLoss()

        self.clip_gradient  = cfg['clip_gradient']
        if self.clip_gradient is not None:
            self.clip_gradient = float(cfg['clip_gradient'])

        # IMPORTANT TO LOAD WEIGHTS
        self.load_weights(self.load_epoch)

        # Training
        if self.training:
            normalize = TSN.IdentityTransform()

            # Dataset
            dataset_kwargs = {'split_set': cfg['train_set']}
            train_tfm   = torchvision.transforms.Compose([
                            TSN.Stack(roll=False),
                            TSN.ToTorchFormatTensor(div=True),
                            normalize])
            if is_train_aug:
                train_tfm   = torchvision.transforms.Compose([
                                train_aug,
                                TSN.Stack(roll=False),
                                TSN.ToTorchFormatTensor(div=True),
                                normalize])
            train_dataset = TSN_Labels(cfg, cfg['train_set'], train_tfm,
                                          random_shift=True, test_mode=False)
            self.train_sampler = None
            shuffle = cfg['shuffle']
            train_kwargs = {'batch_size'  :   int(cfg['batch_size']),
                            'shuffle'     :   shuffle,
                            'num_workers' :   int(cfg['num_workers']),
                            'pin_memory'  :   True}
            self.train_loader = DataLoader(train_dataset,
                                           sampler=self.train_sampler,
                                           **train_kwargs)

            # Validation
            val_kwargs =   {'batch_size'  :   int(cfg['batch_size']),
                            'shuffle'     :   False,
                            'num_workers' :   int(cfg['num_workers']),
                            'pin_memory'  :   True}
            val_tfm   = torchvision.transforms.Compose([
                            TSN.Stack(roll=False),
                            TSN.ToTorchFormatTensor(div=True),
                            normalize])
            if is_train_aug:
                val_tfm   = torchvision.transforms.Compose([
                                train_aug,
                                TSN.Stack(roll=False),
                                TSN.ToTorchFormatTensor(div=True),
                                normalize])
            val_dataset = TSN_Labels(cfg, cfg['val_set'], val_tfm,
                          random_shift=False, test_mode=False)
            self.val_loader = DataLoader(val_dataset,
                                         sampler=None,
                                         **val_kwargs)

            self.top1  = AverageMeter()
            self.top5  = AverageMeter()
        else:
            self.test_segments  = int(cfg['test_segments'])
            self.num_class      = int(cfg['num_class'])
            self.test_crops         = int(cfg['test_crops'])
            if self.test_crops == 1:
                self.cropping = torchvision.transforms.Compose([
                    TSN.GroupScale(self.net.scale_size),
                    TSN.GroupCenterCrop(self.net.input_size)])
            elif self.test_crops == 10:
                self.cropping = torchvision.transforms.Compose([
                    TSN.GroupOverSample(self.net.input_size, self.net.scale_size)])
            else:
                raise ValueError("Only 1 and 10 crops are supported while we \
                                 got {}".format(self.test_crops))
                
            self.output_list    = []
            self.uvd_list       = []
    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, target             = data_load

        img                     = img.cuda()
        target                  = target.cuda()

        _, out                  = self.net(img)
        loss                    = self.loss(out, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient is not None:
            total_norm = clip_grad_norm_(self.net.parameters(),
                                        self.clip_gradient)

        self.optimizer.step()

        loss_dict = {
            'loss'      : loss.item()
        }
        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, target             = data_load

        img                     = img.cuda()
        target                  = target.cuda()
        batch_size              = img.size(0)

        _, out                  = self.net(img)
        prec1, prec5            = self.accuracy(out,
                                                target,
                                                topk=(1,5))

        self.top1.update(prec1.item(), batch_size)
        self.top5.update(prec5.item(), batch_size)

    def get_valid_loss(self):
        val_loss_dict = {
            'Avg Prec@1': self.top1.avg,
            'Avg Prec@5': self.top5.avg,
        }

        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, cfg, split):
        for data_split in split:
            data_set        = data_split + '_set'
            dataset_kwargs  = {'split_set': cfg[data_set]}
            cfg['aug']      = None
            pred_tfm = torchvision.transforms.Compose([
                self.cropping,
                TSN.Stack(roll=False),
                TSN.ToTorchFormatTensor(div=True),
                TSN.GroupNormalize(self.net.input_mean, self.net.input_std)])
            pred_dataset = TSN_Labels(cfg, cfg[data_set], pred_tfm,
                                         random_shift=True, test_mode=True)
            pred_kwargs =  {'batch_size'  :   1,
                            'shuffle'     :   False,
                            'num_workers' :   int(cfg['num_workers']),
                            'pin_memory'  :   True}
            data_loader = DataLoader(pred_dataset,
                                     sampler=None,
                                     **pred_kwargs)

            self.net.eval()
            with torch.no_grad():
                for data_load in tqdm(data_loader):
                    self.predict_step(data_load)
                self.save_predictions(data_split)

            # Hand prediction
            # cfg['aug']      = None
            # dataset         = FPHA_Hand(cfg, cfg['hpo_' + data_set])
            # kwargs          = {'batch_size'     :   1,
            #                    'shuffle'        :   False,
            #                    'num_workers'    :   int(cfg['num_workers']),
            #                    'pin_memory'     :   True}
            # data_loader     = DataLoader(dataset, sampler=None, **kwargs)
            
            # self.net.eval()
            # with torch.no_grad():
            #     for data_load in tqdm(data_loader):
            #         self.hpo_predict_step(data_load)
            #     self.hpo_save_predictions(data_split)            

    def hpo_predict_step(self, data_load):
        img, _                  = data_load
        img                     = img.cuda()
        uvd, _                  = self.net(img)
        pred_uvd, top_idx       = uvd
        pred_uvd                = pred_uvd.squeeze()
        
        W                       = pred_uvd.shape[3]
        H                       = pred_uvd.shape[2]
        D                       = 5
        FT              = torch.FloatTensor
        yv, xv, zv      = torch.meshgrid([torch.arange(H),
                                            torch.arange(W),
                                            torch.arange(D)])
        grid_x          = xv.repeat((21, 1, 1, 1)).type(FT).cuda()
        grid_y          = yv.repeat((21, 1, 1, 1)).type(FT).cuda()
        grid_z          = zv.repeat((21, 1, 1, 1)).type(FT).cuda()

        pred_uvd[:, 0, :, :, :] = (pred_uvd[:, 0, :, :, :] + grid_x)/W
        pred_uvd[:, 1, :, :, :] = (pred_uvd[:, 1, :, :, :] + grid_y)/H
        pred_uvd[:, 2, :, :, :] = (pred_uvd[:, 2, :, :, :] + grid_z)/D
        pred_uvd    = pred_uvd.contiguous().view(21, 3, -1)
        
        self.uvd_list.append(pred_uvd[:, :, top_idx].cpu().numpy())

    def hpo_save_predictions(self, data_split):
        pred_save = "predict_{}_{}_uvd.txt".format(self.load_epoch,
                                                    data_split)
        pred_file = Path(self.data_dir)/self.exp_dir/pred_save
        np.savetxt(pred_file, np.reshape(self.uvd_list, (-1, 63)))

        self.uvd_list              = []

    def predict_step(self, data_load):
        img, target                 = data_load

        img                         = img.cuda()
        target                      = target.cuda()

        _, out                      = self.net(img)
        out                         = out.cpu().numpy()
        # out = out.reshape((self.test_segments, 1, self.num_class))
        out = out.reshape((self.test_crops,
                           self.test_segments,
                           self.num_class))
        out = out.mean(axis=0)        
        out = out.reshape((self.test_segments, 1, self.num_class))
        self.output_list.append((out, target))
        
    def save_predictions(self, data_split):
        # Top-1 mean class accuracy
        video_pred          = [np.argmax(np.mean(x[0], axis=0)) \
                               for x in self.output_list]
        video_labels        = [x[1].item() for x in self.output_list]
        cf                  = confusion_matrix(video_labels, 
                                               video_pred).astype(float)
        cls_cnt             = cf.sum(axis=1)
        cls_hit             = np.diag(cf)
        cls_acc             = cls_hit/(cls_cnt + 1e-8)
        mc_acc              = np.mean(cls_acc)*100
        acc                 = (np.sum(cls_hit)/len(video_pred))*100
        
        print('Mean Class Accuracy {:.02f}%'.format(mc_acc))
        print('Standard Accuracy {:.02f}%'.format(acc))
        
        self.output_list = []

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