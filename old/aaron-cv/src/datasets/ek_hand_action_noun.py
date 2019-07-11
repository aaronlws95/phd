import random
import torch
import os
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image
from tqdm               import tqdm

from src.utils          import IMG, FPHA, DATA_DIR

class EK_Hand_Action_Noun(data.Dataset):
    """ FPHA image GT, hand keypoint GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        self.split_set      = split_set
        # Roots
        root = Path(DATA_DIR)/'EPIC_KITCHENS_2018'
        # Loading
        self.image_tmpl = "img_{:05d}.jpg"
        with open(root/(split_set), 'r') as f:
            img_labels = f.read().splitlines()
        img_path    = [i.split(' ')[0] for i in img_labels]
        length_path = [int(i.split(' ')[1]) for i in img_labels]
        action_id   = [int(i.split(' ')[2]) for i in img_labels]
        noun_id     = [int(i.split(' ')[3]) for i in img_labels]
        
        self.noun_id    = []
        self.action_id  = []
        self.img_path   = []
        for path, length, action, noun in zip(img_path, length_path, action_id, noun_id):
            for idx in range(length):
                self.img_path.append(os.path.join(path, self.image_tmpl.format(idx)))
                self.action_id.append(action)
                self.noun_id.append(noun)
        
        self.shape          = (int(cfg['img_size']), int(cfg['img_size']))
        if cfg['len'] == 'max':
            self.num_data = len(self.img_path)
        else:
            self.num_data = int(cfg['len'])

        self.is_aug         = cfg['aug']
        if self.is_aug:
            self.is_flip    = cfg['flip']
            self.jitter     = float(cfg['jitter'])
            self.hue        = float(cfg['hue'])
            self.sat        = float(cfg['sat'])
            self.exp        = float(cfg['exp'])
            self.rot        = float(cfg['rot'])
        
    def aug(self, img):
        # Image augmentation
        # Rotate
        rot = random.uniform(-self.rot, self.rot)
        new_img = img.rotate(rot)
        # Translate
        new_img, ofs_info = IMG.jitter_img(new_img, self.jitter, self.shape)
        # Flip
        flip = 0
        if self.is_flip:
            new_img, flip = IMG.flip_img(new_img)
        # Distort HSV
        new_img = IMG.distort_image_HSV(new_img, self.hue, self.sat, self.exp)
        
        return new_img

    def __getitem__(self, index):
        """
        Out:
            img     : Input image
        """        
        img         = Image.open(self.img_path[index])
        action_gt   = self.action_id[index]
        noun_gt     = self.noun_id[index]
        
        if self.is_aug:
            img = self.aug(img)
        else:
            img = img.resize(self.shape)

        img = np.asarray(img)
        img = img/255.0
        img = IMG.imgshape2torch(img)
        
        return (img, action_gt, noun_gt)

    def __len__(self):
        return self.num_data