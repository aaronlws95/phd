import os
import sys
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
from torchvision import transforms
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lmdb_utils import *
from utils.image_utils import *
from utils import YOLO_utils as YOLO
from utils import FPHA_utils as FPHA

class YOLOV3Dataset_COCO(data.Dataset): 
    def __init__(self, conf, train_mode, model, deterministic):
        with open(conf["root"], 'r') as file:
            self.img_files = file.read().splitlines()
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.is_train = train_mode and conf["split"] == "train"
        self.img_size = conf["img_size"]
        self.augment = False
        
        if self.is_train:
            self.augment = conf["aug"]["do_aug"]
            if self.augment:
                self.aug_conf = conf["aug"]
                
        self.label_files = [
            x.replace('images', 'labels').replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt').replace('JPEGImages', 'labels')
            for x in self.img_files]

        if conf["len"] == "max":
            self.num_data = len(self.img_files)
        else:
            self.num_data = conf["len"]  

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img)
        
        if self.augment:
            # SV augmentation by 50%
            fraction = 0.50  # must be < 1.0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * self.aug_conf["sat"] + 1
            S *= a
            if a > 1:
                np.clip(S, None, 255, out=S)

            a = (random.random() * 2 - 1) * self.aug_conf["exp"] + 1
            V *= a
            if a > 1:
                np.clip(V, None, 255, out=V)

            img_hsv[:, :, 1] = S  # .astype(np.uint8)
            img_hsv[:, :, 2] = V  # .astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = YOLO.letterbox(img, height=self.img_size)

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()

            x = np.array([x.split() for x in lines], dtype=np.float32)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh

        # Augment image and labels
        if self.augment:
            deg = self.aug_conf["rot_deg"]
            jit = self.aug_conf["jitter"]
            sjit = self.aug_conf["scale_jitter"]
            shear = self.aug_conf["shear"]
            img, labels = YOLO.random_affine(img,
                                            labels, 
                                            degrees=(-deg, deg), 
                                            translate=(jit, jit), 
                                            scale=(1-sjit, 1 + sjit),
                                            shear=(-shear, shear))

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = YOLO.xyxy2xywh(labels[:, 1:5]) / self.img_size

        if self.augment:
            if self.aug_conf["flip"]:
                # random left-right flip
                lr_flip = True
                if lr_flip and random.random() > 0.5:
                    img = np.fliplr(img)
                    if nL:
                        labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img.transpose(2, 0, 1) #to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    # to deal with different lengths of labels_out
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw
    
class YOLOV3Dataset_VOC(data.Dataset):
    def __init__(self, conf, train_mode, model, deterministic):
        super(YOLOV3Dataset_VOC, self).__init__()
        self.conf = conf
        with open(self.conf["root"], 'r') as file:
            self.lines = file.readlines()        
            
        self.is_train = train_mode and conf["split"] == "train"
        self.shape = (conf["img_size"], conf["img_size"])
        self.is_aug = False
        self.deterministic = deterministic
        
        if self.is_train:
            self.is_aug = conf["aug"]["do_aug"]
            if self.is_aug:
                self.aug_conf = conf["aug"]
                
        if conf["len"] == "max":
            self.num_data = len(self.lines)
        else:
            self.num_data = conf["len"]  
        
    def aug(self, img, labels):
        # marvis implementation
        if self.deterministic:
            random.seed(0)
            
        img, ofs_info = jitter_img(img, self.aug_conf["jitter"], self.shape)
        if self.aug_conf["flip"]:
            img, flip = flip_img(img)
        else:
            flip = 0
        img = distort_image_HSV(img, self.aug_conf["hue"], self.aug_conf["sat"], self.aug_conf["exp"])
        new_labels = []

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
            c = box[0]
            x = (jit[0, 0] + jit[1, 0])/2
            y = (jit[0, 1] + jit[1, 1])/2
            w = new_width
            h = new_height
            if flip:
                x = 0.999 - x                
            new_labels.append([c, x, y, w, h])
        new_labels = np.asarray(new_labels)
        return img, new_labels
        
    def __getitem__(self, index):
        imgpath = self.lines[index].rstrip()
        img = Image.open(imgpath).convert('RGB')
        w, h = img.size
        labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
        labels = np.loadtxt(labpath) # class, x_cen, y_cen, width, height
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)
            
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
        
        nL = len(labels)
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        return torch.from_numpy(img), labels_out, imgpath, (h, w)

    def __len__(self):
        return self.num_data
    
    @staticmethod
    # to deal with different lengths of labels_out
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw    
    
class YOLOV3Dataset_FPHA(data.Dataset):
    def __init__(self, conf, train_mode, model, deterministic):
        super(YOLOV3Dataset_FPHA, self).__init__()
        self.conf = conf
        self.keys = get_keys(os.path.join(self.conf["save_prefix"] + "_keys_cache.p"))
        self.bbox_env = None 
            
        self.is_train = train_mode and conf["split"] == "train"
        self.shape = (conf["img_size"], conf["img_size"])
        self.is_aug = False
        self.deterministic = deterministic
        
        if self.is_train:
            self.is_aug = conf["aug"]["do_aug"]
            if self.is_aug:
                self.aug_conf = conf["aug"]
                
        if conf["len"] == "max":
            self.num_data = len(self.lines)
        else:
            self.num_data = conf["len"]  
        
    def aug(self, img, labels):
        # marvis implementation
        if self.deterministic:
            random.seed(0)
            
        img, ofs_info = jitter_img(img, self.aug_conf["jitter"], self.shape)
        if self.aug_conf["flip"]:
            img, flip = flip_img(img)
        else:
            flip = 0
        img = distort_image_HSV(img, self.aug_conf["hue"], self.aug_conf["sat"], self.aug_conf["exp"])
        new_labels = np.empty_like(labels)

        for i, box in enumerate(labels):
            x1 = box[0] - box[2]/2
            y1 = box[1] - box[3]/2
            x2 = box[0] + box[2]/2
            y2 = box[1] + box[3]/2         
            pts = np.asarray([(x1, y1), (x2, y2)])    
            jit = jitter_points(pts, ofs_info)
            new_width = (jit[1, 0] - jit[0, 0])
            new_height =  (jit[1, 1] - jit[0, 1])            
            if new_width < 0.001 or new_height < 0.001:
                continue
            new_labels[i][0] = (jit[0, 0] + jit[1, 0])/2
            new_labels[i][1] = (jit[0, 1] + jit[1, 1])/2
            new_labels[i][2] = new_width
            new_labels[i][3] = new_height
            if flip: 
                new_labels[i][0] = 0.999 - new_labels[i][0]           
        return img, new_labels
        
    def __init_db(self):
        # necessary for loading env into dataloader
        # https://github.com/chainer/chainermn/issues/129
        self.bbox_env = get_env(os.path.join(self.conf["save_prefix"] + "_bbox.lmdb"))        
        
    def __getitem__(self, index):
        if self.bbox_env is None:
            self.__init_db()  
             
        key = self.keys[index]
        labels = read_lmdb_env(key, self.bbox_env, "float32", 4)
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)
        img = Image.open(os.path.join(self.conf["img_dir"], key))
        w, h = img.size
        
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
        
        nL = len(labels)
        labels_out = torch.zeros((nL, 5))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        return torch.from_numpy(img), labels_out, (h, w)

    def __len__(self):
        return self.num_data
    
    @staticmethod
    # to deal with different lengths of labels_out
    def collate_fn(batch):
        img, label, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), hw    