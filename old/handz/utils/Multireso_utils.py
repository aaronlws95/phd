import sys
import os
import numpy as np

from utils.dir import dir_dict
import utils.FPHA as FPHA

DIR = dir_dict["MULTIRESO_DIR"]

REF_Z = 1000
BBSIZE = 260

def get_xyzuvd_bbox(u0=FPHA.X0_COLOR,v0=FPHA.Y0_COLOR):
    bbox_xyz = np.array([(BBSIZE,BBSIZE,REF_Z)])
    bbox_uvd = FPHA.xyz2uvd_color(bbox_xyz)
    bbox_uvd[0,0] = np.ceil(bbox_uvd[0,0] - u0)
    bbox_uvd[0,1] = np.ceil(bbox_uvd[0,1] - v0)
    return bbox_xyz, bbox_uvd

def xyz2normuvd_color(xyz):
    """
    xyz: (-1, 21, 3)
    """
    _, bbox_uvd = get_xyzuvd_bbox()
    center_joint_idx = 3
   
   if len(xyz.shape) == 2:
       xyz = np.expand_dims(xyz, axis=0)
       
    #mean calculation
    normuvd = []
    hand_center_uvd = []
    for xyz_hand_gt in xyz:
        mean_z = xyz_hand_gt[center_joint_idx, 2]
        xyz_hand_gt[:,2] += REF_Z - mean_z
        uvd = xyzuvd.xyz2uvd_color(xyz_hand_gt)
        mean_u = uvd[center_joint_idx, 0]
        mean_v = uvd[center_joint_idx, 1]

        #U
        uvd[:,0] = (uvd[:,0] - mean_u + bbox_uvd[0,0]/2 ) / bbox_uvd[0,0]
        uvd[np.where(uvd[:,0]>1),0]=1
        uvd[np.where(uvd[:,0]<0),0]=0

        #V
        uvd[:,1] =(uvd[:,1] - mean_v + bbox_uvd[0,1]/2) / bbox_uvd[0,1]
        uvd[ np.where(uvd[:,1]>1),1]=1
        uvd[ np.where(uvd[:,1]<0),1]=0

        # Z
        uvd[:,2] = (uvd[:,2] - REF_Z + BBSIZE/2)/BBSIZE    

        normuvd.append(uvd)
        hand_center_uvd.append([mean_u, mean_v, mean_z])
        
    return np.asarray(normuvd), np.asarray(hand_center_uvd)

def normuvd2xyzuvd_color(norm_uvd, hand_center_uvd):
    """
    norm_uvd: (-1, 21, 3)
    """
    _, bbox_uvd = get_xyzuvd_bbox()
    
    mean_u = np.expand_dims(hand_center_uvd[:, 0], axis=-1)
    mean_v = np.expand_dims(hand_center_uvd[:, 1], axis=-1)
    mean_z = np.expand_dims(hand_center_uvd[:, 2], axis=-1)

    uvd_hand = np.empty_like(norm_uvd).astype('float32')
    uvd_hand[..., 0] = norm_uvd[..., 0]*bbox_uvd[0, 0] + mean_u - bbox_uvd[0,0]/2
    uvd_hand[..., 1] = norm_uvd[..., 1]*bbox_uvd[0, 1] + mean_v - bbox_uvd[0,1]/2
    uvd_hand[..., 2] = norm_uvd[..., 2]*BBSIZE + REF_Z - BBSIZE/2
    xyz = FPHA.uvd2xyz_color(uvd_hand)
    xyz[:, :,2] = xyz[:, :,2] - REF_Z + mean_z
    uvd = FPHA.xyz2uvd_color(xyz)
    
    return xyz, uvd