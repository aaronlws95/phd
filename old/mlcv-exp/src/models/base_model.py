import torch
from pathlib import Path
from tqdm import tqdm

from src import ROOT
from src.loggers import Model_Logger, TBX_Logger
from src.datasets import get_dataset, get_dataloader

class Base_Model:
    ''' Base model class '''
    def __init__(self, cfg, mode, load_epoch):
        self.exp_dir            = cfg['exp_dir']
        self.device             = int(cfg['device'])
        self.logger             = Model_Logger()
        self.tbx_logger         = TBX_Logger(Path(ROOT)/self.exp_dir/'log')
        self.load_epoch         = load_epoch
        self.max_epoch          = int(cfg['max_epoch'])
        self.mode               = mode
        self.save_freq          = int(cfg['save_freq'])
        self.val_freq           = int(cfg['val_freq'])
        self.print_freq         = int(cfg['print_freq'])
        
        self.save_ckpt_dir  = Path(ROOT)/self.exp_dir/'ckpt'
        if not self.save_ckpt_dir.is_dir():
            self.save_ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        if mode not in ['train', 'test']:
            raise ValueError('Mode must be train or test')
        
        self.net                = None
        self.dataloader         = None
        self.optimizer          = None
        self.scheduler          = None
        self.train_dataloader   = None
        self.val_dataloader     = None
        self.test_dataloader    = None
        self.pretrain           = None

    # ========================================================
    # TRAINING
    # ========================================================

    def loss():
        raise NotImplementedError()

    def train_step(self, data_load):
        raise NotImplementedError()

    def post_epoch(self):
        pass
    
    def train(self):
        self.net.train()
        batches_per_epoch = len(self.train_dataloader)
        for epoch in range(self.load_epoch, self.max_epoch):
            self.cur_epoch = epoch
            self.scheduler.step()
            # Iterate over all data
            for cur_step, data_load in enumerate(self.train_dataloader):
                loss_dict = self.train_step(data_load)
                # Get current learning rate
                for param_group in self.optimizer.param_groups:
                    cur_lr = param_group['lr']
                # Logging
                if (cur_step + 1)%self.print_freq == 0:
                    self.logger.log_dict['epoch'] = '{:04d}'.format(epoch + 1)
                    self.logger.log_dict['step'] = '{:05d}/{:05d}'.format(cur_step + 1, 
                                                                batches_per_epoch)
                    for key, val in loss_dict.items():
                        self.logger.log_dict[key] = val
                    self.logger.log_dict['lr'] = cur_lr
                    self.logger.log_step(cur_step)
            self.post_epoch()
            # Validation
            if (epoch + 1)%int(self.val_freq) == 0:
                val_loss_dict = self.validate(epoch + 1)
            # tensorboard logging
            if (epoch + 1)%int(self.val_freq) == 0:
                self.tbx_logger.log_dict['loss'] = loss_dict['loss']
                for key, val in val_loss_dict.items():
                    self.tbx_logger.log_dict[key] = val
            self.tbx_logger.add_summary(epoch + 1)
            # save checkpoint
            if (epoch + 1)%int(self.save_freq) == 0:
                self.save_ckpt(epoch + 1)

     # ========================================================
    # VALIDATION
    # ========================================================

    def valid_step(self, data_load):
        raise NotImplementedError()

    def get_valid_loss(self):
        raise NotImplementedError()

    def validate(self, epoch):
        self.net.eval()
        with torch.no_grad():
            for data_load in tqdm(self.val_dataloader):
                self.valid_step(data_load)
            val_loss_dict = self.get_valid_loss()
        self.net.train()
        return val_loss_dict

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_step(self, data_load):
        raise NotImplementedError()
    
    def save_predictions(self, split):
        raise NotImplementedError()

    def predict(self, cfg, split):
        for s in split:
            cfg['aug']          = None
            cfg['batch_size']   = 1
            cfg['shuffle']      = None
            dataloader = get_dataloader(cfg, get_dataset(cfg, s))
            
            self.net.eval()
            with torch.no_grad():
                for data_load in tqdm(dataloader):
                    self.predict_step(data_load)
                self.save_predictions(s)

    # ========================================================
    # SAVING AND LOADING
    # ========================================================
 
    def save_ckpt(self, epoch):
        state = {'epoch': epoch, 
                 'model_state_dict': self.net.state_dict(), 
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict()}
        torch.save(state, self.save_ckpt_dir/f'model_{epoch}.state')
 
    def _load_ckpt(self, load_dir):
        map_loc = 'cuda:' + str(self.device)
        return torch.load(load_dir, map_location=map_loc)

    def load_weights(self):
        if self.load_epoch == 0:
            if self.pretrain:
                load_dir = Path(ROOT)/self.pretrain
                ckpt = self._load_ckpt(load_dir)
                load_dict = ckpt['model_state_dict']
                cur_dict = self.net.state_dict()
                # Filter out unnecessary keys
                pretrained_dict = {k:v for k,v in load_dict.items() 
                                   if k in cur_dict}
                # Overwrite entries in the existing state dict
                cur_dict.update(pretrained_dict)
                # Load the new state dict
                self.net.load_state_dict(cur_dict)
        else:
            load_dir = Path(self.save_ckpt_dir)/'model_{}.state'.format(self.load_epoch)
            ckpt = self._load_ckpt(load_dir)
            model_state_dict = {k.replace('module.', ''):v for k,v in ckpt['model_state_dict'].items()}    
            self.net.load_state_dict(model_state_dict)
            if self.mode == 'train':
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

