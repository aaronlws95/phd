import random
import torch
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image

from src.utils          import LMDB, IMG, FPHA, DATA_DIR

class FPHA_MR_Crop_Normuvd_Hand(data.Dataset):
    """ FPHA cropped hand image GT, normuvd hand keypoint GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        self.split_set      = split_set
        self.img_dir        = cfg['img_dir']
        self.xyz_gt_env     = None
        self.shape          = (int(cfg['img_size']), int(cfg['img_size']))
        keys_path           = Path(DATA_DIR)/(self.split_set + '_keys_cache.p')
        self.keys           = LMDB.get_keys(keys_path)
        self.visual         = cfg['visual']
        
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
                                                rot,
                                                img.size[0]/2,
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
            img_list        : Multireso cropped hand images 
                              [img0, img0/2, img0/4]
            uvd_gt_normuvd  : Normuvd hand keypoints (21, 3)
            hand_center_uvd : Mean values for reverting to xyzuvd [u, v, d]
        """                
        if self.xyz_gt_env is None:
            self.__init_db()  
             
        key     = self.keys[index]
        img     = Image.open(Path(DATA_DIR)/self.img_dir/key)
        img_ori = np.asarray(img)
        xyz_gt  = LMDB.read_lmdb_env(key, self.xyz_gt_env, 'float32', (21, 3))
        uvd_gt  = FPHA.xyz2uvd_color(xyz_gt)
        uvd_gt_normuvd, hand_center_uvd = FPHA.get_normuvd(xyz_gt)
        
        # Crop
        img, _, _, _ = FPHA.crop_hand(np.asarray(img), uvd_gt)
        img = Image.fromarray(img)

        xyz_gt = FPHA.uvd2xyz_color(uvd_gt)

        if self.is_aug:
            img, uvd_gt = self.aug(img, uvd_gt)
        else:
            img = img.resize(self.shape)

        img0 = img
        img1 = img.resize((self.shape[0]//2, self.shape[1]//2))
        img2 = img.resize((self.shape[0]//4, self.shape[1]//4))

        img_list = [img0, img1, img2]
        
        for i in range(len(img_list)):
            img_list[i] = np.asarray(img_list[i])
            img_list[i] = (img_list[i]/255.0)
            img_list[i] = IMG.imgshape2torch(img_list[i])
            
        if self.visual:
            return (img_ori, img_list, uvd_gt_normuvd, hand_center_uvd)
        return (img_list, uvd_gt_normuvd, hand_center_uvd)

    def __len__(self):
        return self.num_data