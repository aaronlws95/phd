import torch
import os
import time
import torchvision
import torch.nn                     as nn
import numpy                        as np
import torch.nn.functional          as F
from pathlib                        import Path
from tqdm                           import tqdm
from torch.utils.data               import DataLoader
from torch.nn.utils                 import clip_grad_norm_
from sklearn.metrics                import confusion_matrix

from src.models                     import Model
from src.utils                      import IMG, TSN
from src.datasets                   import EK_TSN_Labels
from src.components                 import get_scheduler, get_optimizer, \
                                           TSN_EK_YOLOV2_BB_net

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

class TSN_EK_YOLOV2_BB(Model):
    """ Temporal Stream Network with Epic Kitchen (verb and noun output) 
    with YOLOV2 backbone """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        super().__init__(cfg, training, load_epoch, logger, tb_logger)

        # Inference settings
        if not self.training:
            cfg['num_segments']     = 1
            
        self.net            = TSN_EK_YOLOV2_BB_net(cfg)
        self.net            = self.net.cuda()
        
        if self.training:
            # Optimizer and scheduler
            self.optimizer  = get_optimizer(cfg, self.net)
            self.scheduler  = get_scheduler(cfg, self.optimizer)
            
        self.loss           = nn.CrossEntropyLoss()

        crop_size           = self.net.crop_size
        scale_size          = self.net.scale_size
        input_mean          = self.net.input_mean
        input_std           = self.net.input_std
        train_aug           = self.net.get_augmentation()

        lr                  = float(cfg['learning_rate'])
        momentum            = float(cfg['momentum'])
        weight_decay        = float(cfg['decay'])

        self.clip_gradient  = cfg['clip_gradient']
        if self.clip_gradient is not None:
            self.clip_gradient = float(cfg['clip_gradient'])

        # IMPORTANT TO LOAD WEIGHTS
        self.load_weights(self.load_epoch)

        # Training
        if self.training:
            if cfg['modality'] != 'RGBDiff':
                normalize = TSN.GroupNormalize(input_mean, input_std)
            else:
                normalize = TSN.IdentityTransform()

            # Dataset
            dataset_kwargs = {'split_set': cfg['train_set']}
            train_tfm   = torchvision.transforms.Compose([
                            train_aug,
                            TSN.Stack(roll=False),
                            TSN.ToTorchFormatTensor(div=True),
                            normalize])
            train_dataset = EK_TSN_Labels(cfg, cfg['train_set'], train_tfm,
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
                            train_aug,
                            TSN.Stack(roll=False),
                            TSN.ToTorchFormatTensor(div=True),
                            normalize])
            val_dataset = EK_TSN_Labels(cfg, cfg['val_set'], val_tfm,
                          random_shift=False, test_mode=False)
            self.val_loader = DataLoader(val_dataset,
                                         sampler=None,
                                         **val_kwargs)

            self.top1_verb  = AverageMeter()
            self.top5_verb  = AverageMeter()
            self.top1_noun  = AverageMeter()
            self.top5_noun  = AverageMeter()
        else:
            self.test_crops         = int(cfg['test_crops'])
            self.test_segments      = int(cfg['test_segments'])
            self.num_class_verb     = int(cfg['num_class_verb'])
            self.num_class_noun     = int(cfg['num_class_noun'])
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
                
            self.output_verb_list = []
            self.output_noun_list = []
                        
    # ========================================================
    # TRAINING
    # ========================================================

    def train_step(self, data_load):
        img, target_verb, target_noun = data_load

        img                     = img.cuda()
        target_verb             = target_verb.cuda()
        target_noun             = target_noun.cuda()

        out_verb, out_noun      = self.net(img)
        loss_verb               = self.loss(out_verb, target_verb)
        loss_noun               = self.loss(out_noun, target_noun)
        loss                    = loss_verb + loss_noun

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient is not None:
            total_norm = clip_grad_norm_(self.net.parameters(),
                                        self.clip_gradient)
            # if self.debug:
            #     if total_norm > self.clip_gradient:
            #         print("clip gradient: {} with coef {}".format(total_norm,
            #                                                       self.clip_gradient/total_norm))

        self.optimizer.step()

        loss_dict = {
            'loss'      : loss.item(),
            'loss_verb' : loss_verb.item(),
            'loss_noun' : loss_noun.item(),
        }

        return loss_dict

    # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        img, target_verb, target_noun = data_load

        img                     = img.cuda()
        target_verb             = target_verb.cuda()
        target_noun             = target_noun.cuda()
        batch_size              = img.size(0)

        out_verb, out_noun      = self.net(img)
        prec1_verb, prec5_verb  = self.accuracy(out_verb,
                                                target_verb,
                                                topk=(1,5))
        prec1_noun, prec5_noun  = self.accuracy(out_noun,
                                                target_noun,
                                                topk=(1,5))

        self.top1_verb.update(prec1_verb.item(), batch_size)
        self.top5_verb.update(prec5_verb.item(), batch_size)
        self.top1_noun.update(prec1_noun.item(), batch_size)
        self.top5_noun.update(prec5_noun.item(), batch_size)

    def get_valid_loss(self):
        val_loss_dict = {
            'Avg Prec_verb@1': self.top1_verb.avg,
            'Avg Prec_verb@5': self.top5_verb.avg,
            'Avg Prec_noun@1': self.top1_noun.avg,
            'Avg Prec_noun@5': self.top5_noun.avg,
        }

        self.top1_verb = AverageMeter()
        self.top5_verb = AverageMeter()
        self.top1_noun = AverageMeter()
        self.top5_noun = AverageMeter()
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
            pred_dataset = EK_TSN_Labels(cfg, cfg[data_set], pred_tfm,
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

    def predict_step(self, data_load):
        img, target_verb, target_noun   = data_load

        img                             = img.cuda()
        target_verb                     = target_verb.cuda()
        target_noun                     = target_noun.cuda()

        out_verb, out_noun              = self.net(img)
        out_verb                        = out_verb.cpu().numpy()
        out_noun                        = out_noun.cpu().numpy()

        out_verb = out_verb.reshape((self.test_crops,
                                     self.test_segments,
                                     self.num_class_verb))
        out_verb = out_verb.mean(axis=0)
        out_verb = out_verb.reshape((self.test_segments,
                                     1,
                                     self.num_class_verb))

        
        out_noun = out_noun.reshape((self.test_crops,
                                     self.test_segments,
                                     self.num_class_noun))
        out_noun = out_noun.mean(axis=0)
        out_noun = out_noun.reshape((self.test_segments,
                                     1,
                                     self.num_class_noun))
       
        self.output_verb_list.append((out_verb, target_verb))
        self.output_noun_list.append((out_noun, target_noun))
        
    def save_predictions(self, data_split):
        # Top-1 mean class accuracy
        video_pred_verb     = [np.argmax(np.mean(x[0], axis=0)) \
                               for x in self.output_verb_list]
        video_labels_verb   = [x[1].item() for x in self.output_verb_list]
        cf                  = confusion_matrix(video_labels_verb, 
                                               video_pred_verb).astype(float)
        cls_cnt             = cf.sum(axis=1)
        cls_hit             = np.diag(cf)
        cls_acc             = cls_hit/(cls_cnt + 1e-8)
        verb_acc            = np.mean(cls_acc)*100
        
        video_pred_noun     = [np.argmax(np.mean(x[0], axis=0)) \
                               for x in self.output_noun_list]
        video_labels_noun   = [x[1].item() for x in self.output_noun_list]
        cf                  = confusion_matrix(video_labels_noun, 
                                               video_pred_noun).astype(float)
        cls_cnt             = cf.sum(axis=1)
        cls_hit             = np.diag(cf)
        cls_acc             = cls_hit/(cls_cnt + 1e-8)
        noun_acc            = np.mean(cls_acc)*100

        video_pred_action = []
        for v, n in zip(video_pred_verb, video_pred_noun):
            video_pred_action.append((v, n))
        video_labels_action = []
        for v, n in zip(video_labels_verb, video_labels_noun):
            video_labels_action.append((v, n))

        sum = {}
        cnt = {}
        for p, l in zip(video_pred_action, video_labels_action):
            if p not in sum:
                sum[p] = 0
                cnt[p] = 0
            if l not in sum:
                sum[l] = 0
                cnt[l] = 0
            if p == l:
                sum[p] += 1
                cnt[p] += 1
            else:
                cnt[l] += 1
        cls_acc = []
        for k, _ in cnt.items():
            cls_acc.append(sum[k]/(cnt[k] + 1e-8))
        act_acc = np.mean(cls_acc)*100
        
        print('Accuracy_verb {:.02f}%'.format(verb_acc))
        print('Accuracy_noun {:.02f}%'.format(noun_acc))
        print('Accuracy_action {:.02f}%'.format(act_acc))
        
        self.output_verb_list = []
        self.output_noun_list = []

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