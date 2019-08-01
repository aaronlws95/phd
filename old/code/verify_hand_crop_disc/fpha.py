from pathlib import Path
import numpy as np

from utils import ROOT

# ========================================================
# CONSTANTS
# ========================================================

ORI_WIDTH               = 1920
ORI_HEIGHT              = 1080

REORDER_IDX             = [0, 1, 6, 7, 8, 2, 9,
                           10, 11, 3, 12, 13, 14, 4,
                           15, 16, 17, 5, 18, 19, 20]

CAM_EXTR                = [[0.999988496304, -0.00468848412856,
                            0.000982563360594, 25.7],
                           [0.00469115935266, 0.999985218048,
                            -0.00273845880292, 1.22],
                           [-0.000969709653873, 0.00274303671904,
                            0.99999576807, 3.902],
                           [0, 0, 0, 1]]

FOCAL_LENGTH_X_COLOR    = 1395.749023
FOCAL_LENGTH_Y_COLOR    = 1395.749268
X0_COLOR                = 935.732544
Y0_COLOR                = 540.681030

CAM_INTR_COLOR          = [[FOCAL_LENGTH_X_COLOR, 0, X0_COLOR],
                           [0, FOCAL_LENGTH_Y_COLOR, Y0_COLOR],
                           [0, 0, 1]]

FOCAL_LENGTH_X_DEPTH    = 475.065948
FOCAL_LENGTH_Y_DEPTH    = 475.065857
X0_DEPTH                = 315.944855
Y0_DEPTH                = 245.287079

CAM_INTR_DEPTH          = [[FOCAL_LENGTH_X_DEPTH, 0, X0_DEPTH],
                           [0, FOCAL_LENGTH_Y_DEPTH, Y0_DEPTH],
                           [0, 0, 1]]

BBOX_NORMUVD            = [397, 361, 1004.3588]
BBSIZE                  = 260

INV_CAM_EXTR = [[0.99998855624950122256, 0.0046911597684540387191,
                 -0.00096970967236367877683, -25.701645303388132272],
                [-0.0046884842637616731197, 0.99998527559956268165,
                 0.0027430368219501163773, -1.1101913203320408265],
                [0.00098256339938933108913, -0.0027384588555197885184,
                 0.99999576732453258074, -3.9238944436608977969]]

# ========================================================
# XYZ UVD CONVERSION
# ========================================================

def uvd2xyz_depth(skel_uvd):
    """ UVD space to XYZ space for depth """
    skel_xyz        = np.empty_like(skel_uvd).astype("float32")
    fx0             = FOCAL_LENGTH_X_DEPTH
    fy0             = FOCAL_LENGTH_Y_DEPTH
    skel_xyz[...,0] = (skel_uvd[..., 0]-X0_DEPTH)/fx0*skel_uvd[..., 2]
    skel_xyz[...,1] = (skel_uvd[..., 1]-Y0_DEPTH)/fy0*skel_uvd[..., 2]
    skel_xyz[...,2] = skel_uvd[...,2]
    return skel_xyz

def xyz2uvd_depth(skel_xyz):
    """ XYZ space to UVD space for depth """
    skel_uvd = np.empty_like(skel_xyz).astype("float32")
    skel_uvd[..., 0] = X0_DEPTH + \
                    FOCAL_LENGTH_X_DEPTH*(skel_xyz[..., 0]/skel_xyz[..., 2])
    skel_uvd[..., 1] = Y0_DEPTH + \
                    FOCAL_LENGTH_Y_DEPTH*(skel_xyz[..., 1]/skel_xyz[..., 2])
    skel_uvd[..., 2] = skel_xyz[..., 2]
    return skel_uvd

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

def xyz2ccs_color(skel_xyz):
    """ XYZ space to camera coordinate space for color """
    skel_ccs = np.empty_like(skel_xyz).astype("float32")
    skel_ccs[..., 0] = CAM_EXTR[0][0]*skel_xyz[..., 0] + \
                        CAM_EXTR[0][1]*skel_xyz[..., 1] + \
                        CAM_EXTR[0][2]*skel_xyz[..., 2] + CAM_EXTR[0][3]
    skel_ccs[..., 1] = CAM_EXTR[1][0]*skel_xyz[..., 0] + \
                        CAM_EXTR[1][1]*skel_xyz[..., 1] + \
                        CAM_EXTR[1][2]*skel_xyz[..., 2] + CAM_EXTR[1][3]
    skel_ccs[..., 2] = CAM_EXTR[2][0]*skel_xyz[..., 0] + \
                        CAM_EXTR[2][1]*skel_xyz[..., 1] + \
                        CAM_EXTR[2][2]*skel_xyz[..., 2] + CAM_EXTR[2][3]
    return skel_ccs

def ccs2uvd_color(skel_ccs):
    """ Camera coordinate space to UVD for color """
    skel_uvd = np.empty_like(skel_ccs).astype("float32")
    skel_uvd[..., 0] = X0_COLOR + \
                        FOCAL_LENGTH_X_COLOR*(skel_ccs[..., 0]/skel_ccs[..., 2])
    skel_uvd[..., 1] = Y0_COLOR + \
                        FOCAL_LENGTH_Y_COLOR*(skel_ccs[..., 1]/skel_ccs[..., 2])
    skel_uvd[..., 2] = skel_ccs[..., 2]
    return skel_uvd

def uvd2ccs_color(skel_uvd):
    """ UVD space to camera coordinate space for color """
    skel_ccs            = np.empty_like(skel_uvd).astype("float32")
    fx0                 = FOCAL_LENGTH_X_COLOR
    fy0                 = FOCAL_LENGTH_Y_COLOR
    skel_ccs[..., 2]    = skel_uvd[..., 2]
    skel_ccs[..., 0]    = ((skel_uvd[..., 0] - X0_COLOR)/fx0)*skel_uvd[..., 2]
    skel_ccs[..., 1]    = ((skel_uvd[..., 1]- Y0_COLOR)/fy0)*skel_uvd[..., 2]
    return skel_ccs

def ccs2xyz_color(skel_ccs):
    """ Camera coordinate space to XYZ for color """
    skel_xyz = np.empty_like(skel_ccs).astype("float32")
    skel_xyz[..., 0] = INV_CAM_EXTR[0][0]*skel_ccs[..., 0] + \
                        INV_CAM_EXTR[0][1]*skel_ccs[..., 1] + \
                        INV_CAM_EXTR[0][2]*skel_ccs[..., 2] + INV_CAM_EXTR[0][3]
    skel_xyz[..., 1] = INV_CAM_EXTR[1][0]*skel_ccs[..., 0] + \
                        INV_CAM_EXTR[1][1]*skel_ccs[..., 1] + \
                        INV_CAM_EXTR[1][2]*skel_ccs[..., 2] + INV_CAM_EXTR[1][3]
    skel_xyz[..., 2] = INV_CAM_EXTR[2][0]*skel_ccs[..., 0] + \
                        INV_CAM_EXTR[2][1]*skel_ccs[..., 1] + \
                        INV_CAM_EXTR[2][2]*skel_ccs[..., 2] + INV_CAM_EXTR[2][3]
    return skel_xyz

# ========================================================
# GET LABELS
# ========================================================

def get_action_dict():
    action_dict = {}
    action_list_dir = 'First_Person_Action_Benchmark/action_object_info.txt'
    with open(Path(ROOT)/action_list_dir, 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                l = l.split(' ')
                action_dict[int(l[0]) - 1] = l[1]
    return action_dict

def get_obj_dict():
    obj_dict = []
    action_list_dir = 'First_Person_Action_Benchmark/action_object_info.txt'
    with open(Path(ROOT)/action_list_dir, 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                l = l.split(' ')
                obj_dict.append(l[2])
    obj_dict = np.unique(obj_dict)
    obj_dict = {v: k for v, k in enumerate(obj_dict)}
    return obj_dict