import numpy as np
import utils.cam as cam

def xyz2uvd_color(skel_xyz):
    skel_uvd = np.empty_like(skel_xyz).astype('float32')
    ccs_x = cam.CAM_EXTR[0][0]*skel_xyz[..., 0]+cam.CAM_EXTR[0][1]*skel_xyz[..., 1]+cam.CAM_EXTR[0][2]*skel_xyz[..., 2] + cam.CAM_EXTR[0][3]
    ccs_y = cam.CAM_EXTR[1][0]*skel_xyz[..., 0]+cam.CAM_EXTR[1][1]*skel_xyz[..., 1]+cam.CAM_EXTR[1][2]*skel_xyz[..., 2] + cam.CAM_EXTR[1][3]
    ccs_z = cam.CAM_EXTR[2][0]*skel_xyz[..., 0]+cam.CAM_EXTR[2][1]*skel_xyz[..., 1]+cam.CAM_EXTR[2][2]*skel_xyz[..., 2] + cam.CAM_EXTR[2][3]

    skel_uvd[..., 0] = cam.X0_COLOR+cam.FOCAL_LENGTH_X_COLOR*(ccs_x/ccs_z)
    skel_uvd[..., 1] = cam.Y0_COLOR+cam.FOCAL_LENGTH_Y_COLOR*(ccs_y/ccs_z)
    skel_uvd[..., 2] = ccs_z
    return skel_uvd

def uvd2xyz_color(skel_uvd):
    ccs_z = skel_uvd[..., 2]
    ccs_x = ((skel_uvd[..., 0] - cam.X0_COLOR)/cam.FOCAL_LENGTH_X_COLOR)*ccs_z
    ccs_y = ((skel_uvd[..., 1]- cam.Y0_COLOR)/cam.FOCAL_LENGTH_Y_COLOR)*ccs_z

    skel_xyz = np.empty_like(skel_uvd).astype('float32')
    skel_xyz[..., 0] = cam.INV_CAM_EXTR[0][0]*ccs_x+cam.INV_CAM_EXTR[0][1]*ccs_y+cam.INV_CAM_EXTR[0][2]*ccs_z + cam.INV_CAM_EXTR[0][3]
    skel_xyz[..., 1] = cam.INV_CAM_EXTR[1][0]*ccs_x+cam.INV_CAM_EXTR[1][1]*ccs_y+cam.INV_CAM_EXTR[1][2]*ccs_z + cam.INV_CAM_EXTR[1][3]
    skel_xyz[..., 2] = cam.INV_CAM_EXTR[2][0]*ccs_x+cam.INV_CAM_EXTR[2][1]*ccs_y+cam.INV_CAM_EXTR[2][2]*ccs_z + cam.INV_CAM_EXTR[2][3]
    return skel_xyz

def xyz2ccs_color(skel_xyz):
    skel_ccs = np.empty_like(skel_xyz).astype('float32')
    skel_ccs[..., 0] = cam.CAM_EXTR[0][0]*skel_xyz[..., 0]+cam.CAM_EXTR[0][1]*skel_xyz[..., 1]+cam.CAM_EXTR[0][2]*skel_xyz[..., 2] + cam.CAM_EXTR[0][3]
    skel_ccs[..., 1] = cam.CAM_EXTR[1][0]*skel_xyz[..., 0]+cam.CAM_EXTR[1][1]*skel_xyz[..., 1]+cam.CAM_EXTR[1][2]*skel_xyz[..., 2] + cam.CAM_EXTR[1][3]
    skel_ccs[..., 2] = cam.CAM_EXTR[2][0]*skel_xyz[..., 0]+cam.CAM_EXTR[2][1]*skel_xyz[..., 1]+cam.CAM_EXTR[2][2]*skel_xyz[..., 2] + cam.CAM_EXTR[2][3]
    return skel_ccs

def ccs2uvd_color(skel_ccs):
    skel_uvd = np.empty_like(skel_ccs).astype('float32')
    skel_uvd[..., 0] = cam.X0_COLOR+cam.FOCAL_LENGTH_X_COLOR*(skel_ccs[..., 0]/skel_ccs[..., 2])
    skel_uvd[..., 1] = cam.Y0_COLOR+cam.FOCAL_LENGTH_Y_COLOR*(skel_ccs[..., 1]/skel_ccs[..., 2])
    skel_uvd[..., 2] = skel_ccs[..., 2]
    return skel_uvd

def uvd2ccs_color(skel_uvd):
    skel_ccs = np.empty_like(skel_uvd).astype('float32')
    skel_ccs[..., 2] = skel_uvd[..., 2]
    skel_ccs[..., 0] = ((skel_uvd[..., 0] - cam.X0_COLOR)/cam.FOCAL_LENGTH_X_COLOR)*skel_uvd[..., 2]
    skel_ccs[..., 1] = ((skel_uvd[..., 1]- cam.Y0_COLOR)/cam.FOCAL_LENGTH_Y_COLOR)*skel_uvd[..., 2]
    return skel_ccs

def ccs2xyz_color(skel_ccs):
    skel_xyz = np.empty_like(skel_ccs).astype('float32')
    skel_xyz[..., 0] = cam.INV_CAM_EXTR[0][0]*skel_ccs[..., 0]+cam.INV_CAM_EXTR[0][1]*skel_ccs[..., 1]+cam.INV_CAM_EXTR[0][2]*skel_ccs[..., 2] + cam.INV_CAM_EXTR[0][3]
    skel_xyz[..., 1] = cam.INV_CAM_EXTR[1][0]*skel_ccs[..., 0]+cam.INV_CAM_EXTR[1][1]*skel_ccs[..., 1]+cam.INV_CAM_EXTR[1][2]*skel_ccs[..., 2] + cam.INV_CAM_EXTR[1][3]
    skel_xyz[..., 2] = cam.INV_CAM_EXTR[2][0]*skel_ccs[..., 0]+cam.INV_CAM_EXTR[2][1]*skel_ccs[..., 1]+cam.INV_CAM_EXTR[2][2]*skel_ccs[..., 2] + cam.INV_CAM_EXTR[2][3]
    return skel_xyz
