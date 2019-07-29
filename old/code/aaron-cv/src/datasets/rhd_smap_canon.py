import torch
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image

from src.utils          import RHD, DATA_DIR, IMG

class RHD_Smap_Canon(data.Dataset):
    """ Keypoint scoremap GT, canonical xyz GT, rotation matrix GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        self.split_set      = split_set
        self.img_dir        = cfg['img_dir']
        self.anno_all       = RHD.load_annot(DATA_DIR, self.split_set)
        self.shape          = (int(cfg["img_size"]), int(cfg["img_size"]))
        self.visual         = cfg['visual']

        if cfg["len"] == 'max':
            self.num_data = len(self.anno_all)
        else:
            self.num_data = int(cfg["len"])

        self.is_aug         = cfg["aug"]
        if self.is_aug:
            self.is_flip    = cfg["flip"]
            self.jitter     = float(cfg["jitter"])

    def aug(self, img, uvd_gt):
        # Image augmentation
        # Jitter
        new_img, ofs_info = IMG.jitter_img(img, self.jitter, self.shape)
        # Flip
        flip = 0
        if self.is_flip:
            new_img, flip = IMG.flip_img(new_img)
            
        # Point augmentation
        # Jitter
        new_uvd_gt  = IMG.jitter_points(uvd_gt.copy(), ofs_info)
        # Flip
        if flip:
            new_uvd_gt[:, 0]    = 0.999 - new_uvd_gt[:, 0]
            
        new_uvd_gt = new_uvd_gt.astype("float32")
        return new_img, new_uvd_gt

    def __getitem__(self, index):
        """
        Out:
            scoremap        : Hand keypoint scoremaps (21, h, w)
            xyz_gt_canon    : Canonical xyz keypoints (21, 3)
            rot_mat         : Rotation matrix (3, 3)
            kpt_scale       : Keypoint scaling to revert to original (1)
            K               : Intrinsic camera matrix to convert uvd (3, 3)
            kp_coord_xyz21  : Original xyz keypoints for scaling to original
                              (21, 3)
            hand_side       : 1 if right hand, 0 if left hand
        """        
        img                 = Image.open(Path(DATA_DIR,
                                              self.img_dir,
                                              self.split_set,
                                              'color',
                                              '%.5d.png' % index))
        anno                = self.anno_all[index]
        K                   = anno['K']
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
        kp_coord_xyz21      = np.where(cond_left,
                                       kp_coord_xyz_left, kp_coord_xyz_right)
        hand_side           = np.where(np.greater(num_px_left, num_px_right), 
                                       0, # left hand 
                                       1) # right hand
        # Visibility
        keypoint_vis_left   = keypoint_vis[:21]
        keypoint_vis_right  = keypoint_vis[-21:]
        keypoint_vis21      = np.where(cond_left[:, 0],
                                  keypoint_vis_left,
                                  keypoint_vis_right).astype('bool')

        # UV
        uvd_gt              = RHD.xyz2uvd(kp_coord_xyz21, K)

        # Crop
        crop, uvd_gt = RHD.crop_hand(np.asarray(img), uvd_gt)
        crop = Image.fromarray(crop)

        if self.is_aug:
            uvd_gt = IMG.scale_points_WH(uvd_gt, crop.size, (1,1))
            crop, uvd_gt = self.aug(crop, uvd_gt)
            uvd_gt = IMG.scale_points_WH(uvd_gt, (1, 1), self.shape)
        else:
            uvd_gt = IMG.scale_points_WH(uvd_gt, crop.size, self.shape)
            crop = crop.resize(self.shape)

        # Canonical
        xyz_norm, kpt_scale             = RHD.norm_keypoint(kp_coord_xyz21)
        xyz_gt_canon, inv_rot_mat       = RHD.canonical_transform(xyz_norm)
        rot_mat                         = np.linalg.inv(inv_rot_mat)
        xyz_gt_canon                    = np.squeeze(xyz_gt_canon)

        if hand_side == 1:
            xyz_gt_canon[:, 2] = -xyz_gt_canon[:, 2]

        scoremap = RHD.create_multiple_gaussian_map(uvd_gt,
                                                    self.shape,
                                                    valid_vec=keypoint_vis21)
        scoremap = IMG.imgshape2torch(scoremap)

        if self.visual:
            return (scoremap.astype('float32'),
                    xyz_gt_canon.astype('float32'),
                    rot_mat.astype('float32'),
                    uvd_gt.astype('float32'),
                    kpt_scale.astype('float32'),
                    np.asarray(crop).astype('float32'),
                    K.astype('float32'),
                    kp_coord_xyz21.astype('float32'),
                    np.asarray(img).astype('float32'),
                    hand_side.astype('float32'))
        else:
            return (scoremap.astype('float32'),
                    xyz_gt_canon.astype('float32'),
                    rot_mat.astype('float32'),
                    kpt_scale.astype('float32'),
                    K.astype('float32'),
                    kp_coord_xyz21.astype('float32'),
                    keypoint_vis21.astype('float32'),
                    hand_side.astype('float32'))

    def __len__(self):
        return self.num_data