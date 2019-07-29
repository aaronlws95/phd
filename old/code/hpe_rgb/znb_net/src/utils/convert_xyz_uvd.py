import numpy as np

import utils.camera_info as cam

def get_bbox(bbsize,ref_z,u0,v0):
    bbox_xyz = np.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz2uvd_color(bbox_xyz)
    bbox_uvd[0,0] = np.ceil(bbox_uvd[0,0] - u0)
    bbox_uvd[0,1] = np.ceil(bbox_uvd[0,1] - v0)
    return bbox_xyz, bbox_uvd

def xyz2uvd_color(skel_xyz):
    '''
    input: (..., 3)
    output: (..., 3)
    '''
    skel_uvd = np.empty_like(skel_xyz).astype('float32')
    ccs_x = cam.CAM_EXTR[0][0]*skel_xyz[...,0]+cam.CAM_EXTR[0][1]*skel_xyz[...,1]+cam.CAM_EXTR[0][2]*skel_xyz[...,2] + cam.CAM_EXTR[0][3]
    ccs_y = cam.CAM_EXTR[1][0]*skel_xyz[...,0]+cam.CAM_EXTR[1][1]*skel_xyz[...,1]+cam.CAM_EXTR[1][2]*skel_xyz[...,2] + cam.CAM_EXTR[1][3]
    ccs_z = cam.CAM_EXTR[2][0]*skel_xyz[...,0]+cam.CAM_EXTR[2][1]*skel_xyz[...,1]+cam.CAM_EXTR[2][2]*skel_xyz[...,2] + cam.CAM_EXTR[2][3]

    skel_uvd[...,0] = cam.X0_COLOR+cam.FOCAL_LENGTH_X_COLOR*(ccs_x/ccs_z)
    skel_uvd[...,1] = cam.Y0_COLOR+cam.FOCAL_LENGTH_Y_COLOR*(ccs_y/ccs_z)
    skel_uvd[...,2] = ccs_z
    return skel_uvd

def uvd2xyz_color(skel_uvd):
    '''
    input: (..., 3)
    output: (..., 3)
    '''
    ccs_z = skel_uvd[...,2]
    ccs_x = ((skel_uvd[...,0] - cam.X0_COLOR)/cam.FOCAL_LENGTH_X_COLOR)*ccs_z
    ccs_y = ((skel_uvd[...,1]- cam.Y0_COLOR)/cam.FOCAL_LENGTH_Y_COLOR)*ccs_z

    skel_xyz = np.empty_like(skel_uvd).astype('float32')
    skel_xyz[...,0] = cam.INV_CAM_EXTR[0][0]*ccs_x+cam.INV_CAM_EXTR[0][1]*ccs_y+cam.INV_CAM_EXTR[0][2]*ccs_z + cam.INV_CAM_EXTR[0][3]
    skel_xyz[...,1] = cam.INV_CAM_EXTR[1][0]*ccs_x+cam.INV_CAM_EXTR[1][1]*ccs_y+cam.INV_CAM_EXTR[1][2]*ccs_z + cam.INV_CAM_EXTR[1][3]
    skel_xyz[...,2] = cam.INV_CAM_EXTR[2][0]*ccs_x+cam.INV_CAM_EXTR[2][1]*ccs_y+cam.INV_CAM_EXTR[2][2]*ccs_z + cam.INV_CAM_EXTR[2][3]
    return skel_xyz
