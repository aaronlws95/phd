import torch
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image

from src.utils          import LMDB, IMG, DATA_DIR

class FPHA_Bbox(data.Dataset):
    """ FPHA image GT, bounding box GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        
        self.split_set      = split_set
        self.img_dir        = cfg['img_dir']
        self.bbox_env       = None
        self.shape          = (int(cfg["img_size"]), int(cfg["img_size"]))
        keys_path           = Path(DATA_DIR)/(self.split_set + "_keys_cache.p")
        self.keys           = LMDB.get_keys(keys_path)

        if cfg["len"] == 'max':
            self.num_data = len(self.keys)
        else:
            self.num_data = int(cfg["len"])

        self.is_aug         = cfg["aug"]
        if self.is_aug:
            self.is_flip    = cfg["flip"]
            self.jitter     = float(cfg["jitter"])
            self.hue        = float(cfg["hue"])
            self.sat        = float(cfg["sat"])
            self.exp        = float(cfg["exp"])

    def __init_db(self):
        # necessary for loading env into dataloader
        # https://github.com/chainer/chainermn/issues/129
        pth = str(Path(DATA_DIR)/(self.split_set + "_bbox_gt.lmdb"))
        self.bbox_env = LMDB.get_env(pth)
        
    def aug(self, img, bbox_gt):
        # Image augmentation
        # Jitter
        new_img, ofs_info = IMG.jitter_img(img, self.jitter, self.shape)
        # Flip
        flip = 0
        if self.is_flip:
            new_img, flip = IMG.flip_img(new_img)
        new_img = IMG.distort_image_HSV(new_img, self.hue, self.sat, self.exp)
        
        # Point augmentation
        # Jitter
        x1          = bbox_gt[0] - bbox_gt[2]/2
        y1          = bbox_gt[1] - bbox_gt[3]/2
        x2          = bbox_gt[0] + bbox_gt[2]/2
        y2          = bbox_gt[1] + bbox_gt[3]/2
        pts         = np.asarray([(x1, y1), (x2, y2)])
        jit         = IMG.jitter_points(pts, ofs_info)
        new_x_cen   = (jit[0, 0] + jit[1, 0])/2
        new_y_cen   = (jit[0, 1] + jit[1, 1])/2
        new_width   = (jit[1, 0] - jit[0, 0])
        new_height  =  (jit[1, 1] - jit[0, 1])
        new_bbox_gt = np.asarray([new_x_cen, new_y_cen, new_width, new_height])
        # Flip
        if flip:
            new_bbox_gt[0] = 0.999 - new_bbox_gt[0]
        # If invalid set to original
        if new_width < 0.001 or new_height < 0.001:
            new_img     = img.resize(self.shape)
            new_bbox_gt = bbox_gt

        new_bbox_gt = new_bbox_gt.astype("float32")
        return new_img, new_bbox_gt

    def __getitem__(self, index):
        """
        Out:
            img     : Input image
            bbox_gt : Bounding box [x, y, w, h]
        """        
        if self.bbox_env is None:
            self.__init_db()  
             
        key     = self.keys[index]
        bbox_gt = LMDB.read_lmdb_env(key, self.bbox_env, "float32", 4)
        img     = Image.open(Path(DATA_DIR)/self.img_dir/key)

        if self.is_aug:
            img, bbox_gt = self.aug(img, bbox_gt)
        else:
            img = img.resize(self.shape)

        img = np.asarray(img)
        img = img/255.0
        img = IMG.imgshape2torch(img)
        
        return (img, bbox_gt)

    def __len__(self):
        return self.num_data