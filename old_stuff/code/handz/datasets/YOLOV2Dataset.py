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
from utils import YOLO_utils as YOLO
from utils import FPHA_utils as FPHA

class YOLOV2Dataset_VOC(data.Dataset):
    def __init__(self, conf, train_mode, model, deterministic):
        super(YOLOV2Dataset_VOC, self).__init__()
        self.conf = conf
        with open(self.conf["root"], 'r') as file:
            self.lines = file.readlines()        
        
        self.shape = (conf["img_width"], conf["img_height"])
        self.is_train = train_mode and conf["split"] == "train"
        self.deterministic = deterministic

        if self.deterministic:
            random.seed(0)
        
        self.output_imgpath = not train_mode
        
        if self.conf["len"] == "max":
            self.num_data = len(self.lines)
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
        
    def aug(self, img, labels):
        # marvis implementation
        if self.deterministic:
            random.seed(0)
            
        img, ofs_info = jitter_img(img, self.jitter, self.shape)
        if self.is_flip:
            img, flip = flip_img(img)
        else:
            flip = 0
        img = distort_image_HSV(img, self.hue, self.sat, self.exp)
        max_boxes = 50
        new_labels = np.zeros((max_boxes, 5))
        fill_idx = 0
        for i, box in enumerate(labels):
            x1 = box[1] - box[3]/2
            y1 = box[2] - box[4]/2
            x2 = box[1] + box[3]/2
            y2 = box[2] + box[4]/2         
            pts = np.asarray([(x1, y1), (x2, y2)])    
            jit = jitter_points(pts, ofs_info)
            new_width = (jit[1, 0] - jit[0, 0])
            new_height =  (jit[1, 1] - jit[0, 1])            
            if new_width < 0.001 or new_height < 0.001:
                continue
            new_labels[fill_idx][0] = box[0]
            new_labels[fill_idx][1] = (jit[0, 0] + jit[1, 0])/2
            new_labels[fill_idx][2] = (jit[0, 1] + jit[1, 1])/2
            new_labels[fill_idx][3] = new_width
            new_labels[fill_idx][4] = new_height
            if flip:
                new_labels[fill_idx][1] = 0.999 - new_labels[fill_idx][1]  
            fill_idx += 1          
        return img, new_labels

    def aug_plus(self, img, labels):
        #ultralytics implementation
        # add rotation, shearing
        
        img = distort_image_HSV(img, self.hue, self.sat, self.exp)
        
        img = np.asarray(img)
        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=self.shape[1])      
 
        aug_labels = labels.copy().astype("float32")
       
        aug_labels[:, 1] = ratio * w * (labels[:, 1] - labels[:, 3] / 2) + padw
        aug_labels[:, 2] = ratio * h * (labels[:, 2] - labels[:, 4] / 2) + padh
        aug_labels[:, 3] = ratio * w * (labels[:, 1] + labels[:, 3] / 2) + padw
        aug_labels[:, 4] = ratio * h * (labels[:, 2] + labels[:, 4] / 2) + padh

        img, aug_labels = random_affine(img, 
                                        aug_labels, 
                                        degrees=(-self.rot_deg, self.rot_deg), 
                                        translate=(self.jitter, self.jitter), 
                                        scale=(1 - self.scale_jitter, 1 + self.scale_jitter),
                                        shear=(-self.shear, self.shear))
        
        aug_labels[:, 1:5] = YOLO.xyxy2xywh(aug_labels[:, 1:5]) / self.shape[1]

        if self.is_flip and random.random() > 0.5:
            img = np.fliplr(img)
            aug_labels[:, 1] = 1 - aug_labels[:, 1]
            
        max_boxes = 50
        new_labels = np.zeros((max_boxes, 5))          
        new_labels[:len(aug_labels), :] = aug_labels
        return img, new_labels
        
    def __getitem__(self, index):
        imgpath = self.lines[index].rstrip()
        img = Image.open(imgpath).convert('RGB')
        labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
        labels = np.loadtxt(labpath) # class, x_cen, y_cen, width, height
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)
            
        if self.is_train:
            
            if self.is_aug:          
                img, labels = self.aug(img, labels)
            else:
                img = img.resize(self.shape)
                max_boxes = 50
                new_labels = np.zeros((max_boxes, 5))          
                new_labels[:len(labels), :] = labels
                labels = new_labels      
        else:
            img = img.resize(self.shape)
            max_boxes = 50
            new_labels = np.zeros((max_boxes, 5))          
            new_labels[:len(labels), :] = labels                  
            labels = new_labels
            
        img = np.asarray(img)
        img = (img / 255.0)
        img = imgshape2torch(img)
        
        if self.output_imgpath: 
            # output imgpath only when doing prediction
            return (img, imgpath)
        else:
            return (img, labels)

    def __len__(self):
        return self.num_data

class YOLOV2Dataset_FPHA(data.Dataset):
    def __init__(self, conf, train_mode, model, deterministic):
        super(YOLOV2Dataset_FPHA, self).__init__()
        self.conf = conf
        self.keys = get_keys(os.path.join(self.conf["save_prefix"] + "_keys_cache.p"))
        self.bbox_env = None
        
        self.shape = (conf["img_width"], conf["img_height"])
        self.is_train = train_mode and conf["split"] == "train"
    
        if self.conf["len"] == "max":
            self.num_data = len(self.keys)
        else:
            self.num_data = self.conf["len"]   
            
        if self.is_train:                
            self.batch_size = conf["batch_size"]
            self.num_workers = conf["num_workers"]
            self.is_aug = self.conf["aug"]      
            self.is_flip = self.conf["flip"]
                
            if self.is_aug:
                self.jitter = self.conf["jitter"]
                self.hue = self.conf["hue"]
                self.sat = self.conf["sat"] 
                self.exp = self.conf["exp"]   

    def __init_db(self):
        # necessary for loading env into dataloader
        # https://github.com/chainer/chainermn/issues/129
        self.bbox_env = get_env(os.path.join(self.conf["save_prefix"] + "_bbox.lmdb"))
        
    def aug(self, img, labels):
        new_img, ofs_info = jitter_img(img, self.jitter, self.shape)
        if self.is_flip:
            new_img, flip = flip_img(new_img)
        else:
            flip = 0
        new_img = distort_image_HSV(new_img, self.hue, self.sat, self.exp)
        
        x1 = labels[0] - labels[2]/2
        y1 = labels[1] - labels[3]/2
        x2 = labels[0] + labels[2]/2
        y2 = labels[1] + labels[3]/2         
        pts = np.asarray([(x1, y1), (x2, y2)])    
        jit = jitter_points(pts, ofs_info)
        new_x_cen = (jit[0, 0] + jit[1, 0])/2
        new_y_cen = (jit[0, 1] + jit[1, 1])/2        
        new_width = (jit[1, 0] - jit[0, 0])
        new_height =  (jit[1, 1] - jit[0, 1])            

        new_labels = np.asarray([new_x_cen, new_y_cen, new_width, new_height])

        if flip:
            new_labels[0] = 0.999 - new_labels[0]     
            
        if new_width < 0.001 or new_height < 0.001:
            new_img = img.resize(self.shape)
            new_labels = labels                  
          
        return new_img, new_labels.astype("float32")

    def __getitem__(self, index):
        if self.bbox_env is None:
            self.__init_db()  
             
        key = self.keys[index]
        labels = read_lmdb_env(key, self.bbox_env, "float32", 4)

        img = Image.open(os.path.join(self.conf["img_dir"], key))
            
        if self.is_train:

            if self.is_aug:          
                img, labels = self.aug(img, labels)
            else:
                img = img.resize(self.shape)  
        else:
            img = img.resize(self.shape)            
            
        img = np.asarray(img)
        img = (img / 255.0)
        img = imgshape2torch(img)
        
        return (img, labels)

    def __len__(self):
        return self.num_data
    
class YOLOV2Dataset_FPHA_reg(data.Dataset):
    def __init__(self, conf, train_mode, model, deterministic):
        super(YOLOV2Dataset_FPHA_reg, self).__init__()
        self.conf = conf
        self.keys = get_keys(os.path.join(self.conf["save_prefix"] + "_keys_cache.p"))
        self.bbox_env = None
        self.xyz_gt_env = None
        
        self.shape = (conf["img_width"], conf["img_height"])
        self.is_train = train_mode and conf["split"] == "train"
    
        if self.conf["len"] == "max":
            self.num_data = len(self.keys)
        else:
            self.num_data = self.conf["len"]   
            
        if self.is_train:                
            self.batch_size = conf["batch_size"]
            self.num_workers = conf["num_workers"]
            self.is_aug = self.conf["aug"]      
            self.is_flip = self.conf["flip"]
                
            if self.is_aug:
                self.jitter = self.conf["jitter"]
                self.hue = self.conf["hue"]
                self.sat = self.conf["sat"] 
                self.exp = self.conf["exp"]   

    def __init_db(self):
        # necessary for loading env into dataloader
        # https://github.com/chainer/chainermn/issues/129
        self.bbox_env = get_env(os.path.join(self.conf["save_prefix"] + "_bbox.lmdb"))
        self.xyz_gt_env = get_env(os.path.join(self.conf["save_prefix"] + "_xyz_gt.lmdb"))
        
    def aug(self, img, labels, uvd_gt):
        new_img, ofs_info = jitter_img(img, self.jitter, self.shape)
        if self.is_flip:
            new_img, flip = flip_img(new_img)
        else:
            flip = 0
        new_img = distort_image_HSV(new_img, self.hue, self.sat, self.exp)
        
        x1 = labels[0] - labels[2]/2
        y1 = labels[1] - labels[3]/2
        x2 = labels[0] + labels[2]/2
        y2 = labels[1] + labels[3]/2         
        pts = np.asarray([(x1, y1), (x2, y2)])    
        jit = jitter_points(pts, ofs_info)
        new_uvd_gt = jitter_points(uvd_gt.copy(), ofs_info)
        
        new_x_cen = (jit[0, 0] + jit[1, 0])/2
        new_y_cen = (jit[0, 1] + jit[1, 1])/2        
        new_width = (jit[1, 0] - jit[0, 0])
        new_height =  (jit[1, 1] - jit[0, 1])            

        new_labels = np.asarray([new_x_cen, new_y_cen, new_width, new_height])

        if flip:
            new_labels[0] = 0.999 - new_labels[0]     
            new_uvd_gt[:, 0] = 0.999 - new_uvd_gt[:, 0]
            
        if new_width < 0.001 or new_height < 0.001:
            new_img = img.resize(self.shape)
            new_labels = labels                  
            new_uvd_gt = uvd_gt
            
        return new_img, new_labels.astype("float32"), new_uvd_gt.astype("float32")

    def __getitem__(self, index):
        if self.bbox_env or self.xyz_gt_env is None:
            self.__init_db()  
             
        key = self.keys[index]
        labels = read_lmdb_env(key, self.bbox_env, "float32", 4)
        xyz_gt = read_lmdb_env(key, self.xyz_gt_env, "float32", (21, 3))
        uvd_gt = FPHA.xyz2uvd_color(xyz_gt)
        uvd_gt = scale_points_WH(uvd_gt, (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT), (1,1))
        uvd_gt[..., 2] /= FPHA.REF_DEPTH         
        
        img = Image.open(os.path.join(self.conf["img_dir"], key))
            
        if self.is_train:
            if self.is_aug:          
                img, labels, uvd_gt = self.aug(img, labels, uvd_gt)
            else:
                img = img.resize(self.shape)  
        else:
            img = img.resize(self.shape)            
            
        img = np.asarray(img)
        img = (img / 255.0)
        img = imgshape2torch(img)
        
        return (img, labels, uvd_gt)

    def __len__(self):
        return self.num_data    