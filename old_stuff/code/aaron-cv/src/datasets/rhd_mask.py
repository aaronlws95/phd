import torch
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image

from src.utils          import RHD, DATA_DIR, IMG

class RHD_Mask(data.Dataset):
    """ RHD image GT, mask GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        self.split_set      = split_set
        self.img_dir        = cfg['img_dir']
        self.anno_all       = RHD.load_annot(DATA_DIR, self.split_set)
        self.shape          = (int(cfg['img_size']), int(cfg['img_size']))
        
        if cfg['len'] == 'max':
            self.num_data = len(self.anno_all)
        else:
            self.num_data = int(cfg['len'])

        self.is_aug         = cfg['aug']
        if self.is_aug:
            self.is_flip    = cfg['flip']
            self.jitter     = float(cfg['jitter'])
            self.hue        = float(cfg['hue'])
            self.sat        = float(cfg['sat'])
            self.exp        = float(cfg['exp'])
            self.crop       = cfg['crop']
            
    def __getitem__(self, index):
        """
        Out:
            img     : Input image 
            mask    : Hand mask (2, h, w)
        """
        img                 = Image.open(Path(DATA_DIR,
                                              self.img_dir, 
                                              self.split_set,
                                              'color', 
                                              '%.5d.png' % index))
        anno                = self.anno_all[index]
        mask                = Image.open(Path(DATA_DIR,
                                              self.img_dir,
                                              self.split_set,
                                              'mask', 
                                              '%.5d.png' % index))

        if self.is_aug:
            img, mask = IMG.aug_img_mask(img, mask, self.jitter, 
                                         self.is_flip, self.hue, self.sat,
                                         self.exp, self.crop, self.shape)
        else:
            mask = mask.resize(self.shape)
            img = img.resize(self.shape)

        hand_mask           = np.greater(mask, 1)
        bg_mask             = np.logical_not(hand_mask)
        total_mask          = np.stack([bg_mask, 
                                        hand_mask], 2).astype('float32')

        img                 = np.asarray(img)
        img                 = img/255.0
        img                 = IMG.imgshape2torch(img)
        total_mask          = IMG.imgshape2torch(total_mask)
        
        return (img, total_mask)

    def __len__(self):
        return self.num_data