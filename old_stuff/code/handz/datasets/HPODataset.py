import os
import sys
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import random
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lmdb_utils import *
from utils.image_utils import *
from utils import HPO_utils as HPO
from utils import FPHA_utils as FPHA

class HPODataset_FPHA(data.Dataset):
    def __init__(self, conf, train_mode, model, deterministic):
        super(HPODataset_FPHA, self).__init__()
        self.conf = conf
        self.shape = (conf["img_width"], conf["img_height"])
        self.is_train = train_mode and conf["split"] == "train"
        self.keys = get_keys(os.path.join(self.conf["save_prefix"] + "_keys_cache.p"))
        self.xyz_gt_env = None
        
        if self.conf["len"] == "max":
            self.num_data = len(self.keys)
        else:
            self.num_data = self.conf["len"]   
            
        if self.is_train:                
            self.batch_size = conf["batch_size"]
            self.num_workers = conf["num_workers"]
            self.is_aug = self.conf["aug"]      

            if self.is_aug:
                self.jitter = self.conf["jitter"]
                self.hue = self.conf["hue"]
                self.sat = self.conf["sat"] 
                self.exp = self.conf["exp"]   
                self.rot_deg = self.conf["rot_deg"]
                self.scale_jitter = self.conf["scale_jitter"]
                self.is_flip = self.conf["flip"]
                self.shear = self.conf["shear"]


    def __init_db(self):
        # necessary for loading env into dataloader
        # https://github.com/chainer/chainermn/issues/129
        self.xyz_gt_env = get_env(os.path.join(self.conf["save_prefix"] + "_xyz_gt.lmdb"))
        
    def aug(self, img, uvd_gt):
        img, ofs_info = jitter_img(img, self.jitter, self.shape)
        img = distort_image_HSV(img, self.hue, self.sat, self.exp)
        uvd_gt = jitter_points(uvd_gt, ofs_info)
        return img, uvd_gt
    
    def aug_plus(self, img, labels):
        #ultralytics implementation
        # add rotation, shearing
        
        img = distort_image_HSV(img, self.hue, self.sat, self.exp)
        
        img = img.resize(self.shape)
        img = np.asarray(img)
 
        aug_labels = labels.copy().astype("float32")
        aug_labels = scale_points_WH(aug_labels, (1,1), self.shape)
        img, aug_labels[:, :2] = random_affine_pts(img, aug_labels[:, :2], 
                                                   degrees=(-self.rot_deg, self.rot_deg), 
                                                   translate=(self.jitter, self.jitter), 
                                                   scale=(1 - self.scale_jitter, 1 + self.scale_jitter),
                                                   shear=(-self.shear, self.shear))
        aug_labels = scale_points_WH(aug_labels, self.shape, (1,1))
        if self.is_flip and random.random() > 0.5:
            img = np.fliplr(img)
            aug_labels[:, 0] = 1 - aug_labels[:, 0]
            
        return img, aug_labels    
    
    def __getitem__(self, index):
        if self.xyz_gt_env is None:
            self.__init_db()
        
        key = self.keys[index]
        xyz_gt = read_lmdb_env(key, self.xyz_gt_env, "float32", (21, 3))
        uvd_gt = FPHA.xyz2uvd_color(xyz_gt)
        uvd_gt = scale_points_WH(uvd_gt, (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT), (1,1))
        uvd_gt[..., 2] /= FPHA.REF_DEPTH 
        img = Image.open(os.path.join(self.conf["img_dir"], key))
        
        if self.is_train:    
            if self.is_aug:          
                img, uvd_gt = self.aug(img, uvd_gt)
            else:
                img = img.resize(self.shape)
        else:
            img = img.resize(self.shape)
            
        img = np.asarray(img)
        img = (img / 255.0) - 0.5
        img = imgshape2torch(img)
            
        return (img, uvd_gt)

    def __len__(self):
        return self.num_data
