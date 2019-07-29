import os
import sys
import torch
import random
import numpy            as np
import torch.utils.data as data
from PIL                import Image
from torchvision        import transforms
from pathlib            import Path

from src.utils          import DATA_DIR, IMG

class VOC_Bbox(data.Dataset):
    """ Image GT, Bounding box labels GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()

        with open(Path(DATA_DIR)/split_set, 'r') as f:
            self.lines = f.readlines()

        self.shape = (int(cfg['img_size']), int(cfg['img_size']))

        if cfg["len"] == 'max':
            self.num_data = len(self.lines)
        else:
            self.num_data = int(cfg["len"])

        self.is_aug         = cfg["aug"]
        if self.is_aug:
            self.is_flip    = cfg["flip"]
            self.jitter     = float(cfg["jitter"])
            self.hue        = float(cfg["hue"])
            self.sat        = float(cfg["sat"])
            self.exp        = float(cfg["exp"])

    def aug(self, img, labels):
        # Image augmentation
        # Jitter
        img, ofs_info = IMG.jitter_img(img, self.jitter, self.shape)
        # Flip
        flip = 0
        if self.is_flip:
            img, flip = IMG.flip_img(img)
        img = IMG.distort_image_HSV(img, self.hue, self.sat, self.exp)
        
        # Point augmentation
        # Setup
        # Set max boxes to 50 as placeholder for uniform label size
        # Empty labels set to [0, 0, 0, 0, 0]
        max_boxes = 50
        new_labels = np.zeros((max_boxes, 5))
        fill_idx = 0
        for i, box in enumerate(labels):
            x1 = box[1] - box[3]/2
            y1 = box[2] - box[4]/2
            x2 = box[1] + box[3]/2
            y2 = box[2] + box[4]/2
            pts = np.asarray([(x1, y1), (x2, y2)])
            # Jitter
            jit = IMG.jitter_points(pts, ofs_info)
            new_width   = (jit[1, 0] - jit[0, 0])
            new_height  =  (jit[1, 1] - jit[0, 1])
            # If invalid
            if new_width < 0.001 or new_height < 0.001:
                continue
            new_labels[fill_idx][0] = box[0]
            new_labels[fill_idx][1] = (jit[0, 0] + jit[1, 0])/2
            new_labels[fill_idx][2] = (jit[0, 1] + jit[1, 1])/2
            new_labels[fill_idx][3] = new_width
            new_labels[fill_idx][4] = new_height
            # Flip
            if flip:
                new_labels[fill_idx][1] = 0.999 - new_labels[fill_idx][1]
            fill_idx += 1
        return img, new_labels

    def __getitem__(self, index):
        """
        Out:
            img     : Input image
            labels  : Bounding box labels (50, [cls, x, y, w, h])
            imgpath : Path to image used for prediction
        """
        imgpath = self.lines[index].rstrip()
        img     = Image.open(imgpath).convert('RGB')
        
        labpath = imgpath.replace('images', 'labels')
        labpath = labpath.replace('JPEGImages', 'labels')
        labpath = labpath.replace('.jpg', '.txt')
        labpath = labpath.replace('.png','.txt')
        labels  = np.loadtxt(labpath) # class, x_cen, y_cen, width, height
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)

        if self.is_aug:
            img, labels = self.aug(img, labels)
        else:
            img                         = img.resize(self.shape)
            max_boxes                   = 50
            new_labels                  = np.zeros((max_boxes, 5))
            new_labels[:len(labels), :] = labels
            labels                      = new_labels

        img = np.asarray(img)
        img = img/255.0
        img = IMG.imgshape2torch(img)

        return (img, labels, imgpath)

    def __len__(self):
        return self.num_data