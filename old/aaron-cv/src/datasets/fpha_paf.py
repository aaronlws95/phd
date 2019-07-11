import random
import torch
import numpy            as np
import torch.utils.data as data
from pathlib            import Path
from PIL                import Image

from src.utils          import IMG, FPHA, DATA_DIR

def gen_heatmap(center, acc_map, stride=8, sigma=8, in_size=256):
    grid_y = in_size//stride
    grid_x = in_size//stride
    start = stride/2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx*stride + start
    yy = yy*stride + start
    d2 = (xx - center[0])**2 + (yy - center[1])**2
    exponent = d2/2.0/sigma/sigma
    mask = exponent <= 4.6052 # np.log(100) base e
    con_map = np.exp(-exponent)
    con_map = np.multiply(mask, con_map)
    acc_map += con_map
    acc_map[acc_map > 1.0] = 1.0
    return acc_map

class FPHA_PAF(data.Dataset):
    """ FPHA image GT, hand keypoint GT """
    def __init__(self, cfg, split_set=None):
        super().__init__()
        # Roots
        root = Path(DATA_DIR)/'First_Person_Action_Benchmark'
        self.img_root = Path(DATA_DIR)/cfg['img_root']
        # Loading
        with open(root/(split_set + '_img.txt'), 'r') as f:
            self.img_path = f.read().splitlines()
        xyz_gt = np.loadtxt(root/(split_set + '_xyz.txt'))
        self.xyz_gt = np.reshape(xyz_gt, (-1, 21, 3))
        # Length
        if cfg['len'] == 'max':
            self.num_data = len(self.img_path)
        else:
            self.num_data = int(cfg['len'])
        # Params
        self.img_size = int(cfg['img_size'])
        self.stride = int(cfg['stride'])
        
    def __getitem__(self, index):
        img     = Image.open(self.img_root/self.img_path[index])
        xyz     = self.xyz_gt[index]
        uvd     = FPHA.xyz2uvd_color(xyz)
        
        # Visibility mask
        vis = np.zeros(21)
        for i in range(21):
            if uvd[i, 0] <= 0 or uvd[i, 0] >= FPHA.ORI_WIDTH or \
                uvd[i, 1] <=0 or uvd[i, 1] >= FPHA.ORI_HEIGHT:
                    vis[i] = 0
            else:
                vis[i] = 1

        uvd     = IMG.scale_points_WH(uvd, 
                                      (FPHA.ORI_WIDTH, FPHA.ORI_HEIGHT),
                                      (self.img_size, self.img_size))
        
        # Heatmaps
        heatmaps    = np.zeros((self.img_size//self.stride, self.img_size//self.stride, 21))
        heatmap_vis = np.ones((self.img_size//self.stride, self.img_size//self.stride, 21))
        for i in range(21):
            if vis[i] == 0:
                heatmap_vis[:, :, i] = 0
                continue
            center              = uvd[i, :2]
            gaussian_map        = heatmaps[:, :, i]
            heatmaps[:, :, i]   = gen_heatmap(center, gaussian_map,
                                              stride=self.stride, 
                                              in_size=self.img_size)

        img = img.resize((self.img_size, self.img_size))
        img = np.asarray(img)
        img = img/255
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        for i in range(3):
            img[:, :, i] = img[:, :, i] - means[i]
            img[:, :, i] = img[:, :, i]/stds[i]

        img         = img.transpose((2, 0, 1)).astype(np.float32)
        heatmaps    = heatmaps.transpose((2, 0, 1)).astype(np.float32)
        heatmap_vis = heatmap_vis.transpose((2, 0, 1)).astype(np.float32)
        
        return (img, heatmaps, heatmap_vis, uvd)

    def __len__(self):
        return self.num_data