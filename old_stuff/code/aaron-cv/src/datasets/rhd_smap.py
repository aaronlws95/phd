import torch
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image

from src.utils          import RHD, DATA_DIR, IMG

class RHD_Smap(data.Dataset):
    """ RHD cropped image GT, scoremap GT, visibility list GT"""
    def __init__(self, cfg, split_set=None):
        super().__init__()
        self.split_set      = split_set
        self.img_dir        = cfg['img_dir']
        self.anno_all       = RHD.load_annot(DATA_DIR, self.split_set)
        self.shape          = (int(cfg["img_size"]), int(cfg["img_size"]))

        if cfg["len"] == 'max':
            self.num_data = len(self.anno_all)
        else:
            self.num_data = int(cfg["len"])

        self.is_aug         = cfg["aug"]
        if self.is_aug:
            self.is_flip    = cfg["flip"]
            self.jitter     = float(cfg["jitter"])
            self.hue        = float(cfg["hue"])
            self.sat        = float(cfg["sat"])
            self.exp        = float(cfg["exp"])

    def aug(self, img, uvd_gt):
        new_img, ofs_info = IMG.jitter_img(img, self.jitter, self.shape)
        flip = 0
        if self.is_flip:
            new_img, flip = IMG.flip_img(new_img)
        new_img = IMG.distort_image_HSV(new_img, self.hue, self.sat, self.exp)

        new_uvd_gt  = IMG.jitter_points(uvd_gt.copy(), ofs_info)

        if flip:
            new_uvd_gt[:, 0]    = 0.999 - new_uvd_gt[:, 0]

        new_uvd_gt = new_uvd_gt.astype("float32")

        return new_img, new_uvd_gt

    def __getitem__(self, index):
        """
        Out:
            img             : Input cropped hand image
            scoremap        : Hand keypoint scoremaps (21, h, w)
            uvd_gt          : Hand keypoints for validation (21, 3)
            keypoint_vis21  : Keypoint visibility list (21)
        """
        img                 = Image.open(Path(DATA_DIR,
                                              self.img_dir,
                                              self.split_set,
                                              'color',
                                              '%.5d.png' % index))
        anno                = self.anno_all[index]
        keypoint_xyz        = anno['xyz']
        keypoint_vis        = anno['uv_vis'][:, 2]
        keypoint_uv         = anno['uv_vis'][:, :2]
        mask                = Image.open(Path(DATA_DIR,
                                              self.img_dir,
                                              self.split_set,
                                              'mask',
                                              '%.5d.png' % index))
        mask                = np.asarray(mask)

        # Find dominant hand
        cond_l              = np.logical_and(np.greater(mask, 1),
                                             np.less(mask, 18))
        cond_r              = np.greater(mask, 17)
        hand_map_l          = np.where(cond_l, 1, 0)
        hand_map_r          = np.where(cond_r, 1, 0)
        num_px_left         = np.sum(hand_map_l)
        num_px_right        = np.sum(hand_map_r)
        kp_coord_xyz_left   = keypoint_xyz[:21, :]
        kp_coord_xyz_right  = keypoint_xyz[-21:, :]
        cond_left           = np.logical_and(np.ones(kp_coord_xyz_left.shape),
                                             np.greater(num_px_left,
                                                        num_px_right))

        # Visibility
        keypoint_vis_left   = keypoint_vis[:21]
        keypoint_vis_right  = keypoint_vis[-21:]
        keypoint_vis21      = np.where(cond_left[:, 0],
                                  keypoint_vis_left,
                                  keypoint_vis_right).astype('bool')

        # UV
        keypoint_uv_left    = keypoint_uv[:21, :]
        keypoint_uv_right   = keypoint_uv[-21:, :]
        uvd_gt              = np.where(cond_left[:, :2],
                                       keypoint_uv_left,
                                       keypoint_uv_right)

        # Crop
        img, uvd_gt = RHD.crop_hand(np.asarray(img), uvd_gt)
        img = Image.fromarray(img)

        if self.is_aug:
            uvd_gt = IMG.scale_points_WH(uvd_gt, img.size, (1,1))
            img, uvd_gt = self.aug(img, uvd_gt)
            uvd_gt = IMG.scale_points_WH(uvd_gt, (1, 1), self.shape)
        else:
            uvd_gt = IMG.scale_points_WH(uvd_gt, img.size, self.shape)
            img = img.resize(self.shape)

        scoremap = RHD.create_multiple_gaussian_map(uvd_gt,
                                                    self.shape,
                                                    valid_vec=keypoint_vis21)
        scoremap = IMG.imgshape2torch(scoremap)

        img = np.asarray(img)
        img = img/255.0
        img = IMG.imgshape2torch(img)

        return (img,
                scoremap.astype('float32'),
                uvd_gt,
                keypoint_vis21.astype('float32'))

    def __len__(self):
        return self.num_data