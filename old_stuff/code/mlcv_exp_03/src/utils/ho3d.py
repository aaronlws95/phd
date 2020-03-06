from pathlib import Path
import numpy as np

from src import ROOT

# ========================================================
# CONSTANTS
# ========================================================

ORI_WIDTH       = 640
ORI_HEIGHT      = 480

FOCAL_LENGTH_X  = 617.343
FOCAL_LENGTH_Y  = 617.343
X0              = 312.42
Y0              = 241.42

CAM_INTR        = [[FOCAL_LENGTH_X, 0, X0],
                  [0, FOCAL_LENGTH_Y, Y0],
                  [0, 0, 1]]

# ========================================================
# XYZ UVD CONVERSION
# ========================================================

def from_opengl_coord(skel_ccs):
    # ho3d annotations are in OpenGL coordinate system
    coordChangeMat = np.array([[-1., 0., 0.], [0, 1., 0.], [0., 0., 1.]], dtype=np.float32)
    skel_ccs = skel_ccs.dot(coordChangeMat.T)
    return skel_ccs

def ccs2uvd(skel_ccs):
    """ Camera coordinate space to UVD for color """
    skel_uvd = np.empty_like(skel_ccs).astype("float32")
    skel_uvd[..., 0] = X0 + \
                        FOCAL_LENGTH_X*(skel_ccs[..., 0]/skel_ccs[..., 2])
    skel_uvd[..., 1] = Y0 + \
                        FOCAL_LENGTH_Y*(skel_ccs[..., 1]/skel_ccs[..., 2])
    skel_uvd[..., 2] = skel_ccs[..., 2]
    return skel_uvd

def uvd2ccs(skel_uvd):
    """ UVD space to camera coordinate space for color """
    skel_ccs            = np.empty_like(skel_uvd).astype("float32")
    fx0                 = FOCAL_LENGTH_X
    fy0                 = FOCAL_LENGTH_Y
    skel_ccs[..., 2]    = skel_uvd[..., 2]
    skel_ccs[..., 0]    = ((skel_uvd[..., 0] - X0)/fx0)*skel_uvd[..., 2]
    skel_ccs[..., 1]    = ((skel_uvd[..., 1]- Y0)/fy0)*skel_uvd[..., 2]
    return skel_ccs