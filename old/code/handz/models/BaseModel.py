import os
import torch.nn as nn
import torch

class BaseModel():
    def __init__(self, conf, device, train_mode, exp_dir, deterministic, logger=None):
        self.conf               = conf
        self.device             = device
        self.deterministic      = deterministic
        self.logger             = logger
        self.save_dir           = os.path.join(exp_dir)
        self.save_ckpt_dir      = os.path.join(self.save_dir, 'ckpt')
        self.save_freq          = conf["save_freq"]
        self.name               = conf["name"]
        self.train_mode         = train_mode
        
        if self.logger:
            for key, val in conf.items():
                if key == 'name':
                    logger.log('NETWORK: ' + val)
                elif key != 'optimizer' or key != 'scheduler' or key != 'pretrain':
                    logger.log(key.upper() + ': ' + str(val))
    
    def init_network(self, net, load_epoch):
        self.net = net
        if self.train_mode:
            self.optimizer = self.get_optimizer(self.conf["optimizer"])
            self.scheduler = self.get_scheduler(self.conf["scheduler"])
        
        if not os.path.exists(self.save_ckpt_dir):
            os.makedirs(self.save_ckpt_dir)        
        
        if load_epoch != 0 :
            load_dir = os.path.join(self.save_ckpt_dir, f"model_{load_epoch}.state")
            ckpt = self.get_ckpt(load_dir)
            if self.logger:
                self.logger.log(f"LOADED CHECKPOINT: {load_dir}")
            self.load_ckpt(self.net, ckpt["model_state_dict"])
            if self.train_mode:
                self.load_ckpt(self.optimizer, ckpt["optimizer_state_dict"])
                if self.scheduler:
                    self.load_ckpt(self.scheduler, ckpt["scheduler_state_dict"])
        
        if self.device != "cpu":
            # if gpu
            self.net.cuda()
            if self.train_mode:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            # if distributed
            if self.device == "distributed":
                net = nn.parallel.DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])        
    
    def get_ckpt(self, load_dir):            
        if self.device != "distributed":
            if self.device == "cpu":
                map_loc = self.device
            else:
                map_loc = "cuda:" + str(self.device)          
            return torch.load(load_dir, map_location=map_loc)
        else:
            return torch.load(load_dir)
    def load_ckpt(self, comp, state_dict):
        comp.load_state_dict(state_dict)
        
    def get_optimizer(self, conf):
        lr = conf["learning_rate"]
        name = conf["name"]
        momentum = conf["momentum"]
        decay = conf["decay"]        
        if self.logger:
            self.logger.log('---------')
            self.logger.log('OPTIMIZER')
            self.logger.log('---------')
            for key, val in conf.items():
                self.logger.log(key.upper() + ': ' + str(val))
        if name == "Adam":
            return torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=decay)
        elif name == "Adam_yolo":
            """
            Adam but scale lr and weight_decay by batch_size
            """
            batch_size = conf["batch_size"]            
            params_dict = dict(self.net.named_parameters())
            params = []
            for key, value in params_dict.items(): 
                if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                    params += [{'params': [value], 'weight_decay': 0.0}]
                else:
                    params += [{'params': [value], 'weight_decay': decay*batch_size}]                     
            return torch.optim.Adam(self.net.parameters(), lr=lr/batch_size, weight_decay=decay*batch_size)        
        elif name == "SGD":
            return torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, dampening=0, weight_decay=decay)
        elif name == "SGD_yolo":
            """
            SGD as in https://github.com/marvis/pytorch-yolo2/blob/master/train.py
            divide by batch_size so the gradient doesn't explode
            """
            batch_size = conf["batch_size"]
            params_dict = dict(self.net.named_parameters())
            params = []
            for key, value in params_dict.items(): 
                if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                    params += [{'params': [value], 'weight_decay': 0.0}]
                else:
                    params += [{'params': [value], 'weight_decay': decay*batch_size}]            
            return torch.optim.SGD(self.net.parameters(), lr=lr/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)        
        else:
            raise ValueError(f"{name} is not a valid optimizer")

    def get_scheduler(self, conf):
        if conf["is_schedule"]: 
            
            if conf["name"] == "multistep":
                milestones = conf["milestones"]
                gamma = conf["gamma"]
                if self.logger:
                    self.logger.log(f"SCHEDULER: {milestones} {gamma}")
                return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                            milestones=milestones, 
                                                            gamma=gamma)
            else:
                raise ValueError(f"{name} is not a valid scheduler")
        else:
            return None
        
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
                 "optimizer_state_dict": self.optimizer.state_dict()}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.save_ckpt_dir, f"model_{epoch}.state"))
    
    def init_epoch(self, epoch):
        pass