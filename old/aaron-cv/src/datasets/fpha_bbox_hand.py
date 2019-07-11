import torch
import random
import torch.utils.data as data
import numpy            as np
from pathlib            import Path
from PIL                import Image

from src.utils          import LMDB, IMG, FPHA, DATA_DIR


class FPHA_Bbox_Hand(data.Dataset):
    """ FPHA image GT, bounding box GT, hand keypoint GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        self.split_set      = split_set
        self.img_dir        = cfg['img_dir']
        self.bbox_env       = None
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
        pth = str(Path(DATA_DIR)/(self.split_set + '_bbox_gt.lmdb'))
        self.bbox_env = LMDB.get_env(pth)
        pth = str(Path(DATA_DIR)/(self.split_set + '_xyz_gt.lmdb'))
        self.xyz_gt_env = LMDB.get_env(pth)

    def aug(self, img, bbox_gt, uvd_gt):
        # Image augmentation
        # Rotate
        rot     = random.uniform(-self.rot, self.rot)
        new_img = img.rotate(rot)
        # Translate
        new_img, ofs_info = IMG.jitter_img(new_img, self.jitter, self.shape)
        # Flip
        flip = 0
        if self.is_flip:
            new_img, flip = IMG.flip_img(new_img)
        # Distory HSV
        new_img = IMG.distort_image_HSV(new_img, self.hue, self.sat, self.exp)

        # Bbox augmentation
        # Rotate
        x1          = bbox_gt[0] - bbox_gt[2]/2
        y1          = bbox_gt[1] - bbox_gt[3]/2
        x2          = bbox_gt[0] + bbox_gt[2]/2
        y2          = bbox_gt[1] + bbox_gt[3]/2
        pts         = np.asarray([x1*img.size[0],
                                  y1*img.size[1],
                                  x2*img.size[0],
                                  y2*img.size[1]])
        pts         = np.expand_dims(pts, 0)
        pts         = IMG.update_rotate_bbox(pts,
                                             rot,
                                             img.size[0]/2,
                                             img.size[1]/2,
                                             img.size[0],
                                             img.size[1])
        pts         = np.squeeze(pts)
        pts         = np.asarray([(pts[0]/img.size[0],
                                   pts[1]/img.size[1]),
                                  (pts[2]/img.size[0],
                                   pts[3]/img.size[1])])
        # Translate
        jit         = IMG.jitter_points(pts, ofs_info)
        new_x_cen   = (jit[0, 0] + jit[1, 0])/2
        new_y_cen   = (jit[0, 1] + jit[1, 1])/2
        new_width   = jit[1, 0] - jit[0, 0]
        new_height  = jit[1, 1] - jit[0, 1]
        new_bbox_gt = np.asarray([new_x_cen, new_y_cen, new_width, new_height])

        # Hand augmentation
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
        new_uvd_gt = IMG.jitter_points(new_uvd_gt, ofs_info)
        
        # Flip
        if flip:
            new_bbox_gt[0]      = 0.999 - new_bbox_gt[0]
            new_uvd_gt[:, 0]    = 0.999 - new_uvd_gt[:, 0]

        # If augmentation is invalid revert to original
        if new_width < 0.001 or new_height < 0.001:
            new_img     = img.resize(self.shape)
            new_bbox_gt = bbox_gt
            new_uvd_gt  = uvd_gt

        new_bbox_gt = new_bbox_gt.astype('float32')
        new_uvd_gt  = new_uvd_gt.astype('float32')
        return new_img, new_bbox_gt, new_uvd_gt

    def __getitem__(self, index):
        """
        Out:
            img     : Input image
            bbox_gt : Bounding box [x, y, w, h]
            uvd_gt  : Hand keypoints (21, 3)
        """
        if self.bbox_env or self.xyz_gt_env is None:
            self.__init_db()

        key     = self.keys[index]
        img     = Image.open(Path(DATA_DIR)/self.img_dir/key)
        bbox_gt = LMDB.read_lmdb_env(key, self.bbox_env, 'float32', 4)
        xyz_gt  = LMDB.read_lmdb_env(key, self.xyz_gt_env, 'float32', (21, 3))
        uvd_gt  = FPHA.xyz2uvd_color(xyz_gt)
        uvd_gt  = IMG.scale_points_WH(uvd_gt,
                                      (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT), (1,1))
        uvd_gt[..., 2] /= FPHA.REF_DEPTH

        if self.is_aug:
            img, bbox_gt, uvd_gt = self.aug(img, bbox_gt, uvd_gt)
        else:
            img = img.resize(self.shape)

        img = np.asarray(img)
        img = img/255.0
        img = IMG.imgshape2torch(img)

        return (img, bbox_gt, uvd_gt)

    def __len__(self):
        return self.num_data