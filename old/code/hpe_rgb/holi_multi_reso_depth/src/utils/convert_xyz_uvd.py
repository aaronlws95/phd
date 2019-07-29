import numpy as np

import utils.camera_info as cam

def get_bbox(bbsize,ref_z,u0,v0):
    bbox_xyz = np.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz2uvd_depth(bbox_xyz)
    bbox_uvd[0,0] = np.ceil(bbox_uvd[0,0] - u0)
    bbox_uvd[0,1] = np.ceil(bbox_uvd[0,1] - v0)
    return bbox_xyz, bbox_uvd

def xyz2uvd_color(skel_xyz):
    '''
    input: (..., 3)
    output: (..., 3)
    '''
    skel_uvd = np.empty_like(skel_xyz).astype('float32')
    ccs_x = cam_EXTR[0,0]*skel_xyz[...,0]+cam_EXTR[0,1]*skel_xyz[...,1]+cam_EXTR[0,2]*skel_xyz[...,2] + cam_EXTR[0,3]
    ccs_y = cam_EXTR[1,0]*skel_xyz[...,0]+cam_EXTR[1,1]*skel_xyz[...,1]+cam_EXTR[1,2]*skel_xyz[...,2] + cam_EXTR[1,3]
    ccs_z = cam_EXTR[2,0]*skel_xyz[...,0]+cam_EXTR[2,1]*skel_xyz[...,1]+cam_EXTR[2,2]*skel_xyz[...,2] + cam_EXTR[2,3]

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
    skel_xyz[...,0] = cam.INV_CAM_EXTR[0,0]*ccs_x+cam.INV_CAM_EXTR[0,1]*ccs_y+cam.INV_CAM_EXTR[0,2]*ccs_z + cam.INV_CAM_EXTR[0,3]
    skel_xyz[...,1] = cam.INV_CAM_EXTR[1,0]*ccs_x+cam.INV_CAM_EXTR[1,1]*ccs_y+cam.INV_CAM_EXTR[1,2]*ccs_z + cam.INV_CAM_EXTR[1,3]
    skel_xyz[...,2] = cam.INV_CAM_EXTR[2,0]*ccs_x+cam.INV_CAM_EXTR[2,1]*ccs_y+cam.INV_CAM_EXTR[2,2]*ccs_z + cam.INV_CAM_EXTR[2,3]
    return skel_xyz

def uvd2xyz_depth(skel_uvd):
    skel_xyz = np.empty_like(skel_uvd).astype('float32')
    skel_xyz[...,0] = (skel_uvd[...,0] - cam.X0_DEPTH)/cam.FOCAL_LENGTH_X_DEPTH*skel_uvd[...,2]
    skel_xyz[...,1] = (skel_uvd[...,1]- cam.Y0_DEPTH)/cam.FOCAL_LENGTH_Y_DEPTH*skel_uvd[...,2]
    skel_xyz[...,2] = skel_uvd[...,2]
    return skel_xyz

def xyz2uvd_depth(skel_xyz):
    skel_uvd = np.empty_like(skel_xyz).astype('float32')
    skel_uvd[...,0] = cam.X0_DEPTH + cam.FOCAL_LENGTH_X_DEPTH*(skel_xyz[...,0]/skel_xyz[...,2])
    skel_uvd[...,1] = cam.Y0_DEPTH +  cam.FOCAL_LENGTH_Y_DEPTH*(skel_xyz[...,1]/skel_xyz[...,2])
    skel_uvd[...,2] = skel_xyz[...,2]
    return skel_uvd

def normuvd2xyzuvd_depth(norm_uvd,hand_center_uvd):
    #works for (batch_size, -1, 3)
    u0 = cam.X0_DEPTH
    v0 = cam.Y0_DEPTH
    bbsize = 260
    ref_z = 1000
    mean_u = np.expand_dims(hand_center_uvd[:, 0], axis=-1)
    mean_v = np.expand_dims(hand_center_uvd[:, 1], axis=-1)
    mean_z = np.expand_dims(hand_center_uvd[:, 2], axis=-1)

    _, bbox_uvd = get_bbox(bbsize, ref_z, u0, v0)

    uvd_hand = np.empty_like(norm_uvd).astype('float32')
    uvd_hand[..., 0] = norm_uvd[..., 0]*bbox_uvd[0, 0] + mean_u - bbox_uvd[0,0]/2
    uvd_hand[..., 1] = norm_uvd[..., 1]*bbox_uvd[0, 1] + mean_v - bbox_uvd[0,1]/2
    uvd_hand[..., 2] = norm_uvd[..., 2]*bbsize + ref_z - bbsize/2
    xyz = uvd2xyz_depth(uvd_hand)
    xyz[:, :,2] = xyz[:, :,2] - ref_z + mean_z
    uvd = xyz2uvd_depth(xyz)
    return xyz, uvd
