import random
import torch
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image

from src.utils          import LMDB, IMG, FPHA, DATA_DIR

class FPHA_Hand(data.Dataset):
    """ FPHA image GT, hand keypoint GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        self.split_set      = split_set
        self.img_dir        = cfg['img_dir']
        self.xyz_gt_env     = None
        self.shape          = (int(cfg['img_size']), int(cfg['img_size']))
        keys_path           = Path(DATA_DIR)/(self.split_set + '_keys_cache.p')
        self.keys           = LMDB.get_keys(keys_path)
        
        if cfg['len'] == 'max':
            self.num_data = len(self.keys)
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
            
    def __init_db(self):
        # necessary for loading env into dataloader
        # https://github.com/chainer/chainermn/issues/129
        pth = str(Path(DATA_DIR)/(self.split_set + '_xyz_gt.lmdb'))
        self.xyz_gt_env = LMDB.get_env(pth)
        
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
        if self.xyz_gt_env is None:
            self.__init_db()  
             
        key     = self.keys[index]
        img     = Image.open(Path(DATA_DIR)/self.img_dir/key)
        xyz_gt  = LMDB.read_lmdb_env(key, self.xyz_gt_env, 'float32', (21, 3))
        uvd_gt  = FPHA.xyz2uvd_color(xyz_gt)
        uvd_gt  = IMG.scale_points_WH(uvd_gt, 
                                      (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT), (1,1))
        uvd_gt[..., 2] /= FPHA.REF_DEPTH

        if self.is_aug:
            img, uvd_gt = self.aug(img, uvd_gt)
        else:
            img = img.resize(self.shape)

        img = np.asarray(img)
        img = img/255.0
        img = IMG.imgshape2torch(img)
        
        return (img, uvd_gt)

    def __len__(self):
        return self.num_data