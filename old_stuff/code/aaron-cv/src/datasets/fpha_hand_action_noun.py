import random
import torch
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image

from src.utils          import IMG, FPHA, DATA_DIR

class FPHA_Hand_Action_Noun(data.Dataset):
    """ FPHA image GT, hand keypoint GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        self.split_set      = split_set
        # Roots
        root = Path(DATA_DIR)/'First_Person_Action_Benchmark'
        self.img_root = Path(DATA_DIR)/cfg['img_root']
        # Loading
        with open(root/(split_set + '_img.txt'), 'r') as f:
            img_labels = f.read().splitlines()
        self.img_path   = [i.split(' ')[0] for i in img_labels]
        self.action_id  = [int(i.split(' ')[1]) for i in img_labels]
        self.noun_id    = [int(i.split(' ')[2]) for i in img_labels]
        xyz_gt          = np.loadtxt(root/(split_set + '_xyz.txt'))
        self.xyz_gt     = np.reshape(xyz_gt, (-1, 21, 3))
        
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
        
    def aug(self, img, uvd_gt):
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

        # Point augmentation
        # Rotate
        new_uvd_gt          = uvd_gt.copy()
        new_uvd_gt          = IMG.scale_points_WH(new_uvd_gt,
                                                  (1,1), 
                                                  img.size)
        new_uvd_gt[:, :2]   = IMG.rotate_points(new_uvd_gt[:, :2],
                                                rot,img.size[0]/2,
                                                img.size[1]/2,
                                                img.size[0],
                                                img.size[1])
        new_uvd_gt          = IMG.scale_points_WH(new_uvd_gt,
                                                  img.size,
                                                  (1,1))
        # Translate
        new_uvd_gt  = IMG.jitter_points(new_uvd_gt, ofs_info)
        # Flip
        if flip:
            new_uvd_gt[:, 0]    = 0.999 - new_uvd_gt[:, 0]

        new_uvd_gt = new_uvd_gt.astype('float32')
        return new_img, new_uvd_gt

    def __getitem__(self, index):
        """
        Out:
            img     : Input image
            uvd_gt  : Hand keypoints (21, 3)
        """        
        img         = Image.open(self.img_root/self.img_path[index])
        xyz_gt      = self.xyz_gt[index]
        uvd_gt      = FPHA.xyz2uvd_color(xyz_gt)
        uvd_gt      = IMG.scale_points_WH(uvd_gt, 
                                          (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT), (1,1))
        uvd_gt[..., 2] /= FPHA.REF_DEPTH
        action_gt   = self.action_id[index]
        noun_gt     = self.noun_id[index]
        
        if self.is_aug:
            img, uvd_gt = self.aug(img, uvd_gt)
        else:
            img = img.resize(self.shape)

        img = np.asarray(img)
        img = img/255.0
        img = IMG.imgshape2torch(img)
        
        return (img, uvd_gt, action_gt, noun_gt)

    def __len__(self):
        return self.num_data