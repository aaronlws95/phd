from pathlib import Path
import numpy as np

from src import ROOT

ORI_WIDTH       = 640
ORI_HEIGHT      = 480

FOCAL_LENGTH_X_COLOR  = 617.173
FOCAL_LENGTH_Y_COLOR  = 617.173
X0_COLOR              = 315.453
Y0_COLOR              = 242.259

CAM_INTR        = [[FOCAL_LENGTH_X_COLOR, 0, X0_COLOR],
                  [0, FOCAL_LENGTH_Y_COLOR, Y0_COLOR],
                  [0, 0, 1]]

CAM_EXTR        = [[1.0000, 0.00090442, -0.0074, 20.2365],
                   [-0.00071933, 0.9997, 0.0248, 1.2846],
                   [0.0075, -0.0248, 0.9997, 5.7360],
                   [0, 0, 0, 1]]

INV_CAM_EXTR = [[0.99994383406268400144, -0.00072057729688056620185,
                 0.0074196805932044609911, -20.276997032296552807],
                [0.00090504967101100041744 , 0.99968422257296814617,
                 -0.024792909225011632112, -1.1602972626699822687],
                [-0.0074793773368301062326, 0.024805014553902384988,
                 0.99962937705988864471, -5.6143822090347018249]]

def xyz2uvd_color(skel_xyz):
    """ XYZ space to UVD space for color """
    skel_uvd = np.empty_like(skel_xyz).astype("float32")
    ccs_x = CAM_EXTR[0][0]*skel_xyz[..., 0] + \
            CAM_EXTR[0][1]*skel_xyz[..., 1] + \
            CAM_EXTR[0][2]*skel_xyz[..., 2] + CAM_EXTR[0][3]
    ccs_y = CAM_EXTR[1][0]*skel_xyz[..., 0] + \
            CAM_EXTR[1][1]*skel_xyz[..., 1] + \
            CAM_EXTR[1][2]*skel_xyz[..., 2] + CAM_EXTR[1][3]
    ccs_z = CAM_EXTR[2][0]*skel_xyz[..., 0] + \
            CAM_EXTR[2][1]*skel_xyz[..., 1] + \
            CAM_EXTR[2][2]*skel_xyz[..., 2] + CAM_EXTR[2][3]

    skel_uvd[..., 0] = X0_COLOR+FOCAL_LENGTH_X_COLOR*(ccs_x/ccs_z)
    skel_uvd[..., 1] = Y0_COLOR+FOCAL_LENGTH_Y_COLOR*(ccs_y/ccs_z)
    skel_uvd[..., 2] = ccs_z
    return skel_uvd

def uvd2xyz_color(skel_uvd):
    """ UVD space to XYZ space for color """
    ccs_z = skel_uvd[..., 2]
    ccs_x = ((skel_uvd[..., 0]-X0_COLOR)/FOCAL_LENGTH_X_COLOR)*ccs_z
    ccs_y = ((skel_uvd[..., 1]-Y0_COLOR)/FOCAL_LENGTH_Y_COLOR)*ccs_z

    skel_xyz = np.empty_like(skel_uvd).astype("float32")
    skel_xyz[..., 0] = INV_CAM_EXTR[0][0]*ccs_x + \
                        INV_CAM_EXTR[0][1]*ccs_y + \
                        INV_CAM_EXTR[0][2]*ccs_z + INV_CAM_EXTR[0][3]
    skel_xyz[..., 1] = INV_CAM_EXTR[1][0]*ccs_x + \
                        INV_CAM_EXTR[1][1]*ccs_y + \
                        INV_CAM_EXTR[1][2]*ccs_z + INV_CAM_EXTR[1][3]
    skel_xyz[..., 2] = INV_CAM_EXTR[2][0]*ccs_x + \
                        INV_CAM_EXTR[2][1]*ccs_y + \
                        INV_CAM_EXTR[2][2]*ccs_z + INV_CAM_EXTR[2][3]
    return skel_xyz