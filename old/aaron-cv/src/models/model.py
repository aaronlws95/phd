import torch
import torch.nn         as nn
import numpy            as np
from pathlib            import Path
from tqdm               import tqdm

from src.components     import get_optimizer, get_scheduler, Network
from src.datasets       import get_dataloader, get_dataset
from src.utils          import DATA_DIR

class Model:
    """ Base model from which all would inherit """
    def __init__(self, cfg, training, load_epoch, logger, tb_logger):
        self.debug          = cfg['debug']
        
        if self.debug:
            # Deterministic
            torch.cuda.manual_seed(0)
            torch.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(0)

        self.data_dir       = DATA_DIR
        self.exp_dir        = cfg['exp_dir']
        self.training       = training
        self.device         = cfg['device']
        self.load_epoch     = load_epoch
        self.time           = cfg['time']
        self.save_ckpt_dir  = Path(self.data_dir)/self.exp_dir/'ckpt'

        if not self.save_ckpt_dir.is_dir():
            self.save_ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.training:
                self.pretrain       = cfg['pretrain']
                self.val_freq       = int(cfg['val_freq'])
                self.save_freq      = int(cfg['save_freq'])
                self.print_freq     = int(cfg['print_freq'])
                self.max_epoch      = int(cfg['max_epoch'])
                self.logger         = logger
                self.tb_logger      = tb_logger

        # Set net_cfg to false to leave network config to inherited model
        if cfg['net_cfg']:
            cur_path            = Path(__file__).resolve().parents[2] 
            net_cfgfile         = cur_path/'net_cfg'/(cfg['net_cfg'] + '.cfg')
            self.net            = Network(net_cfgfile).cuda()

            if self.training:
                self.train_sampler  = None
                self.val_loader     = None
                self.train_loader   = None
                self.optimizer  = get_optimizer(cfg, self.net)
                self.scheduler  = get_scheduler(cfg, self.optimizer)
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
        else:
            self.net            = None
            if self.training:
                self.train_sampler  = None
                self.val_loader     = None
                self.train_loader   = None
                self.optimizer      = None
                self.scheduler      = None
    # ========================================================
    # TO BE IMPLEMENTED BY INHERITED
    # ========================================================            
            
    def train_step():
        raise NotImplementedError()
    
    def valid_step():
        raise NotImplementedError()

    def get_valid_loss():
        raise NotImplementedError()

    def predict_step():
        raise NotImplementedError()
    
    def save_predictions():
        raise NotImplementedError()
    
    # ========================================================
    # TRAINING
    # ========================================================

    def train(self, rank):
        self.net.train()
        batches_per_epoch = len(self.train_loader)
        for epoch in range(self.load_epoch, self.max_epoch):
            if self.device == "distributed":
                self.train_sampler.set_epoch(epoch)
            if self.scheduler:
                self.scheduler.step()
            # Iterate over all data
            for cur_step, data_load in enumerate(self.train_loader):
                loss_dict = self.train_step(data_load)
                # Get current learning rate
                for param_group in self.optimizer.param_groups:
                    cur_lr = param_group["lr"]
                # Logging
                if (cur_step + 1)%self.print_freq == 0 and rank <= 0:
                    self.logger.log_dict["epoch"] = epoch + 1
                    self.logger.log_dict["step"] = "{}/{}".format(cur_step + 1, 
                                                             batches_per_epoch)
                    for key, val in loss_dict.items():
                        self.logger.log_dict[key] = val
                    self.logger.log_dict["lr"] = cur_lr
                    self.logger.log_step(cur_step)
            # Validation and TB logging
            if rank <= 0:
                self.post_epoch_process(epoch, loss_dict)
                
    def post_epoch_process(self, epoch, loss_dict):
        # validation
        if (epoch+1)%int(self.val_freq) == 0:
            val_loss = self.validate(epoch+1)
        # tensorboard logging
        self.tb_logger.log_dict["loss"] = loss_dict["loss"]
        if (epoch+1)%int(self.val_freq) == 0:
            for key, val in val_loss.items():
                self.tb_logger.log_dict[key] = val
        self.tb_logger.update_scalar_summary(epoch+1)
        # save checkpoint
        if (epoch+1)%int(self.save_freq) == 0:
            self.save_ckpt(epoch+1)

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self, epoch):
        self.net.eval()
        with torch.no_grad():
            for data_load in tqdm(self.val_loader):
                self.valid_step(data_load)
            val_loss = self.get_valid_loss()
            self.net.train()
        return val_loss

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, cfg, split):
        for data_split in split:
            data_set        = data_split + '_set'
            dataset_kwargs  = {'split_set': cfg[data_set]}
            cfg['aug']      = None
            dataset         = get_dataset(cfg, dataset_kwargs)
            sampler         = None
            kwargs          = {'batch_size'     :   int(cfg['batch_size']),
                               'shuffle'        :   False,
                               'num_workers'    :   int(cfg['num_workers']),
                               'pin_memory'     :   True}
            data_loader     = get_dataloader(dataset, sampler, kwargs)        
            
            self.net.eval()
            with torch.no_grad():
                for data_load in tqdm(data_loader):
                    self.predict_step(data_load)
                self.save_predictions(data_split)

    # ========================================================
    # SAVING AND LOADING
    # ========================================================

    def save_ckpt(self, epoch):
        if isinstance(self.net, nn.parallel.DistributedDataParallel):
            network = self.net.module
        else:
            network = self.net
        model_state_dict = network.state_dict()
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        state = {"epoch": epoch, 
                 "model_state_dict": model_state_dict, 
                 "optimizer_state_dict": self.optimizer.state_dict()}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, self.save_ckpt_dir / f"model_{epoch}.state")

    def load_ckpt(self, load_dir):
        if self.device == "distributed":
            return torch.load(load_dir)
        map_loc = "cuda:" + str(self.device)
        return torch.load(load_dir, map_location=map_loc)

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
            load_dir = Path(self.save_ckpt_dir)/f'model_{load_epoch}.state'
            ckpt = self.load_ckpt(load_dir)
            self.net.load_state_dict(ckpt['model_state_dict'])
            if self.training:
                self.optimizer.load_state_dict(
                    ckpt['optimizer_state_dict'])
                if self.scheduler:
                    self.scheduler.load_state_dict(
                        ckpt['scheduler_state_dict'])
 