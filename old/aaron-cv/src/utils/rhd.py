import os
import pickle
import cv2
import numpy                                        as np
import matplotlib.patches                           as patches
from tqdm                   import tqdm
from PIL                    import Image
from matplotlib             import pyplot           as plt
from mpl_toolkits.mplot3d   import Axes3D
from scipy                  import ndimage

from src.utils              import IMG, DATA_DIR

# ========================================================
# CONSTANTS
# ========================================================

REORDER_IDX = [0, 20, 19, 18, 17, 16, 15,
               14, 13, 12, 11, 10, 9, 8,
               7, 6, 5, 4, 3, 2, 1]
ORI_WIDTH   = 320
ORI_HEIGHT  = 320

# ========================================================
# VISUALISATION
# ========================================================

def visualize_joints_2d(ax, joints, joint_idxs=False,
                        links=None, alpha=1, c=None):
    """
    Draw 2d skeleton on matplotlib axis
    Args:
        joints : Hand keypoints (21, 2++)
    """
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha, c=c)

def _draw2djoints(ax, annots, links, alpha=1, c=None):
    """ Draw segments """
    if c:
        colors = [c, c, c, c, c]
    else:
        colors = ["r", "m", "b", "c", "g"]
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)

def _draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1):
    """ Draw segment of given color """
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)

def draw_bbox(ax, box, img_dim, c='r'):
    """
    Draw bounding box
    Args:
        boxes   : [x_cen, y_cen, w, h]
        img_dim : (W, H)
        c       : Fix color of bounding box
    """
    x1      = (box[0] - box[2]/2.0)*img_dim[0]
    y1      = (box[1] - box[3]/2.0)*img_dim[1]
    width   = box[2]*img_dim[0]
    height  = box[3]*img_dim[1]

    rect    = patches.Rectangle((x1, y1), width, height,
                                linewidth=2, edgecolor=c, facecolor='none')
    ax.add_patch(rect)

# ========================================================
# GET GROUND TRUTH
# ========================================================

def get_all_anno(split_set):
    """
    Get annotations for RHD dataset
    Args:
        split_set   : [training, evaluation]
    Out:
        Hand output is for dominant hand side
        xyz         : Hand xyz keypoints (21, 3)
        uv          : Hand uv keypoints (21, 2)
        vis         : Hand visibility (21)
        K           : Intrinsic camera matrix (3, 3)
        hand side   : Hand side (1)
    """
    anno_all    = load_annot(DATA_DIR, split_set)
    K           = []
    xyz         = []
    uv          = []
    vis         = []
    hand_side   = []
    for sample_id, anno in tqdm(anno_all.items()):
        # Get current sample
        keypoint_xyz    = anno['xyz']
        keypoint_vis    = anno['uv_vis'][:, 2]
        # keypoint_uv     = anno['uv_vis'][:, :2]
        mask            = np.asarray(Image.open(os.path.join(DATA_DIR,
                                                        'RHD_published_v2',
                                                        split_set,
                                                        'mask',
                                                        '%.5d.png'
                                                        % sample_id)))
        # Find the dominant hand
        cond_l              = np.logical_and(np.greater(mask, 1),
                                             np.less(mask, 18))
        cond_r              = np.greater(mask, 17)
        hand_map_l          = np.where(cond_l, 1, 0)
        hand_map_r          = np.where(cond_r, 1, 0)
        num_px_left    = np.sum(hand_map_l)
        num_px_right   = np.sum(hand_map_r)
        # XYZ
        kp_coord_xyz_left   = keypoint_xyz[:21, :]
        kp_coord_xyz_right  = keypoint_xyz[-21:, :]
        cond_left           = np.logical_and(np.ones(kp_coord_xyz_left.shape),
                                             np.greater(num_px_left,
                                                        num_px_right))
        kp_coord_xyz21      = np.where(cond_left,
                                       kp_coord_xyz_left,
                                       kp_coord_xyz_right)
        # Hand side
        cur_hand_side       = np.where(np.greater(num_px_left, num_px_right), 
                                       0, # left hand 
                                       1) # right hand        
        # Vis
        keypoint_vis_left   = keypoint_vis[:21]
        keypoint_vis_right  = keypoint_vis[-21:]
        keypoint_vis21      = np.where(cond_left[:, 0],
                                       keypoint_vis_left,
                                       keypoint_vis_right).astype(bool)
        # UV
        keypoint_uv21       = xyz2uvd(kp_coord_xyz21, anno['K'])
        
        xyz.append(kp_coord_xyz21)
        uv.append(keypoint_uv21)
        vis.append(keypoint_vis21)
        K.append(anno['K'])
        hand_side.append(cur_hand_side)

    return xyz, uv, vis, K, hand_side

def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits*2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

def load_annot(dataset_dir, data_split):
    """ Load annotations """
    with open(os.path.join(dataset_dir, 'RHD_published_v2', data_split,
                           'anno_%s.pickle' % data_split), 'rb') as fi:
        anno_all = pickle.load(fi)
    return anno_all

def crop_hand(img, uvd_gt, pad=10):
    """
    Get cropped image and cropped hand from image
    Args:
        img         : (320, 320, 3)
        uvd_gt      : Hand points in img space (21, 2++)
    Out:
        crop        : Cropped image
        uvd_gt_crop : [x_cen, y_cen, w, h]
    """
    x_max = int(np.amax(uvd_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt[:,1])) - pad, 0)

    # Ensure bbox is within image bounds
    if y_max > ORI_HEIGHT:
        y_max = ORI_HEIGHT
    if y_min < 0:
        y_min = 0
    if x_max > ORI_WIDTH:
        x_max = ORI_WIDTH
    if x_min < 0:
        x_min = 0

    crop                = img[y_min:y_max, x_min:x_max, :]
    uvd_gt_crop         = uvd_gt.copy()
    uvd_gt_crop[:, 0]   = uvd_gt_crop[:, 0] - x_min
    uvd_gt_crop[:, 1]   = uvd_gt_crop[:, 1] - y_min
    return crop, uvd_gt_crop

def revert_hand_annot(img, uvd_gt, new_uvd_gt, new_shape,
                      img_size=(ORI_WIDTH, ORI_HEIGHT), pad=10):
    """
    Revert hand annotation to pre-crop location
    Args:
        img             : (320, 320, 3)
        uvd_gt          : Hand points in img space (21, 2++)
        new_uvd_gt      : Cropped uvd_gt (21, 2++)
        new_shape       : Shape of resized image i.e. (256, 256)
        img_size        : Original image shape
    Out:
        uvd_gt_revert   : Reverted hand points (21, 2++)
    """
    x_max = int(np.amax(uvd_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt[:,1])) - pad, 0)

    # ensure bbox is within img bounds
    if y_max > img_size[1]:
        y_max = img_size[1]
    if y_min < 0:
        y_min = 0
    if x_max > img_size[0]:
        x_max = img_size[0]
    if x_min < 0:
        x_min = 0

    crop = img[y_min:y_max, x_min:x_max, :]

    uvd_gt_revert = new_uvd_gt.copy()
    uvd_gt_revert = IMG.scale_points_WH(uvd_gt_revert,
                                        new_shape,
                                        (crop.shape[1], crop.shape[0]))
    uvd_gt_revert[:, 0] = uvd_gt_revert[:, 0] + x_min
    uvd_gt_revert[:, 1] = uvd_gt_revert[:, 1] + y_min

    return uvd_gt_revert

def get_bbox(uvd_gt, img_size=(ORI_WIDTH, ORI_HEIGHT), pad=10):
    """
    Get normalized bounding box for hand
    Args:
        uvd_gt          : Hand points in img space (21, 2++)
    Out:
        bbox            : [x_cen ,y_cen, w, h]
    """
    x_max = int(np.amax(uvd_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt[:,1])) - pad, 0)

    # ensure bbox is within img bounds
    if y_max > img_size[1]:
        y_max = img_size[1]
    if y_min < 0:
        y_min = 0
    if x_max > img_size[0]:
        x_max = img_size[0]
    if x_min < 0:
        x_min = 0

    width   = (x_max - x_min)/img_size[0]
    height  = (y_max - y_min)/img_size[1]
    x_cen   = ((x_min + x_max)/2)/img_size[0]
    y_cen   = ((y_min + y_max)/2)/img_size[1]

    return np.asarray([x_cen, y_cen, width, height])

def create_multiple_gaussian_map(uvd_gt, output_size,
                                 sigma=25.0, valid_vec=None):
    """
    Create Gaussian scoremaps
    Args:
        uvd_gt      : Hand keypoints (21, 2++)
        output_size : Size of scoremap (x, y)
        sigma       : Std dev of Gaussian distribution
        valid_vec   : Visibility vector (21)
    Out:
        scoremap    : Scoremap for each keypoint (x, y, 21)
    """
    coords_uv = np.stack([uvd_gt[:, 1], uvd_gt[:, 0]], -1)
    coords_uv = coords_uv.astype(np.int32)

    if valid_vec is not None:
        valid_vec   = np.squeeze(valid_vec)
        cond_val    = np.greater(valid_vec, 0.5)
    else:
        cond_val    = np.ones(coords_uv[:, 0].shape)
        cond_val    = np.greater(cond_val, 0.5)

    cond_1_in   = np.logical_and(np.less(coords_uv[:, 0], output_size[0]-1),
                                 np.greater(coords_uv[:, 0], 0))
    cond_2_in   = np.logical_and(np.less(coords_uv[:, 1], output_size[1]-1),
                                 np.greater(coords_uv[:, 1], 0))
    cond_in     = np.logical_and(cond_1_in, cond_2_in)
    cond        = np.logical_and(cond_val, cond_in)
    coords_uv   = coords_uv.astype(np.float32)

    # Create meshgrid
    x_range = np.expand_dims(np.arange(output_size[0]), 1)
    y_range = np.expand_dims(np.arange(output_size[1]), 0)

    X       = np.tile(x_range, [1, output_size[1]]).astype(np.float32)
    Y       = np.tile(y_range, [output_size[0], 1]).astype(np.float32)

    X       = np.expand_dims(X, -1)
    Y       = np.expand_dims(Y, -1)

    X_b     = np.tile(X, [1, 1, coords_uv.shape[0]])
    Y_b     = np.tile(Y, [1, 1, coords_uv.shape[0]])

    X_b     -= coords_uv[:, 0]
    Y_b     -= coords_uv[:, 1]

    dist        = np.square(X_b) + np.square(Y_b)
    scoremap    = np.exp(-dist/np.square(sigma))*cond.astype(np.float32)
    return scoremap

def left_handed_rot_mat(dim, angle):
    """ Calculate the left handed rotation matrix """
    #https://butterflyofdream.wordpress.com/2016/07/05/converting-rotation-matrices-of-left-handed-coordinate-system/
    c = np.cos(angle)
    s = np.sin(angle)
    if dim == 'x':
        rot_mat = [[1., 0, 0],
                   [0, c, s],
                   [0, -s, c]]
    elif dim == 'y':
        rot_mat = [[c, 0, -s],
                  [0, 1, 0],
                  [s, 0, c]]
    elif dim == 'z':
        rot_mat = [[c, s, 0],
                   [-s, c, 0],
                   [0, 0, 1]]
    else:
        raise ValueError('dim needs to be x, y or z')

    return rot_mat

def detect_keypoints_from_scoremap(smap):
    """ Get keypoints from scoremaps
    Args:
        smap: (x, y, 21)
    Out:
        keypoint_coords: Hand keypoints (21, 2)
    """
    s = smap.shape
    keypoint_coords = np.zeros((s[2], 2))
    for i in range(s[2]):
        v, u = np.unravel_index(np.argmax(smap[:, :, i]), (s[0], s[1]))
        keypoint_coords[i, 0] = u
        keypoint_coords[i, 1] = v
    return keypoint_coords

def get_keypoint_scale(coords, middle1=11, middle_root=12):
    """ Get keypoint scaling for canonical transformation
    Args:
        coords: (b, 21, 3)
    Out:
        L2 distance from middle1 to middle_root keypoints
    """
    return np.sqrt(np.sum(np.square(coords[:, middle1, :] - \
                                    coords[:, middle_root, :]), axis=-1))

def norm_keypoint(coords, root=0, middle1=11, middle_root=12):
    """ Get normalized keypoint
    Args:
        coords              : (b, 21, 3)
    Out:
        coord_norm          : Normalized scaled coordinate (b, 21, 3)
        root_bone_length    : Keypoint scale (to revert to original form)
    """
    if len(coords.shape) == 2:
        coords_proc = np.expand_dims(coords, axis=0)
    else:
        coords_proc = coords.copy()
    
    ROOT_NODE_ID        = root
    root_bone_length    = get_keypoint_scale(coords_proc)
    coords_norm         = coords_proc/root_bone_length
    translate           = coords_norm[:, ROOT_NODE_ID, :]
    coords_norm         = coords_norm - translate
        
    return coords_norm, root_bone_length

def canonical_transform(coords, root=0, middle=12, pinky=20):
    """ Get canonical transformation
    Args:
        coords          : Output from norm_keypoint (b, 21, 3)
    Out:
        coord_normed    : Canonical coordinate (b, 21, 3)
        total_rot_mat   : Rotation matrix (3, 3)
    """
    ROOT_NODE_ID = root
    ALIGN_NODE_ID = middle # middle root
    ROT_NODE_ID = pinky # pinky root

    # 1. Translate the whole set s.t. the root kp is located in the origin
    # norm, _ = norm_keypoint(coords)

    # 2. Rotate and scale keypoints such that the root bone
    # is of unit length and aligned with the y axis
    p = coords[:, ALIGN_NODE_ID, :] # Thats the point we want on (0, 1, 0)
    # Rotate point into the yz-plane
    alpha = np.arctan2(p[:, 0], p[:, 1])
    rot_mat = left_handed_rot_mat('z', alpha)
    rot_mat = np.asarray(rot_mat).astype('float32')
    coords_t_r1 = np.matmul(coords, rot_mat)
    total_rot_mat = rot_mat

    # Rotate point within the yz-plane onto the xy-plane
    p = coords_t_r1[:, ALIGN_NODE_ID, :]
    beta = -np.arctan2(p[:, 2], p[:, 1])
    rot_mat = left_handed_rot_mat('x', beta + 3.141592653589793)
    rot_mat = np.asarray(rot_mat).astype('float32')
    coords_t_r2 = np.matmul(coords_t_r1, rot_mat)
    total_rot_mat = np.matmul(total_rot_mat, rot_mat)

    # 3. Rotate keypoints such that rotation along the y-axis is defined
    p = coords_t_r2[:, ROT_NODE_ID, :]
    gamma = np.arctan2(p[:, 2], p[:, 0])
    rot_mat = left_handed_rot_mat('y', gamma)
    rot_mat = np.asarray(rot_mat).astype('float32')
    coords_normed = np.matmul(coords_t_r2, rot_mat)
    total_rot_mat = np.matmul(total_rot_mat, rot_mat)

    return coords_normed, total_rot_mat

# ========================================================
# CONVERSION
# ========================================================

def xyxy2xywh(x):
    """"
    Convert box type xyxy to xywh
    Args:
        x: (-1, 4) [x_min, y_min, x_max, y_max]
    Out:
        y: (-1, 4) [x_cen, y_cen, w, h]
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) \
        else np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def xywh2xyxy(x):
    """"
    Convert box type xywh to xyxy
    Args:
        x: (-1, 4) [x_cen, y_cen, w, h]
    Out:
        y: (-1, 4) [x_min, y_min, x_max, y_max]
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) \
        else np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyz2uvd(kp_coord_xyz21, K):
    """
    Convert from XYZ space to UVD space
    Args:
        kp_coord_xyz21  : XYZ keypoints (21, 3)
        K               : Intrinsic camera matrix (3, 3)
    Out:
        kp_coord_uv_proj: UVD keypoints (21, 3)
    """
    kp_coord_uv_proj = np.matmul(kp_coord_xyz21, np.transpose(K))
    kp_coord_uv_proj[:, :2] = kp_coord_uv_proj[:, :2]/kp_coord_uv_proj[:, 2:]
    return kp_coord_uv_proj

def uvd2xyz(kp_coord_uv_proj, K):
    """
    Convert from XYZ space to UVD space
    Args:
        kp_coord_uv_proj    : UVD keypoints (21, 3)
        K                   : Intrinsic camera matrix (3, 3)
    Out:
        kp_coord_xy_proj    : XYZ keypoints (21, 3)
    """
    kp_coord_xy_proj = kp_coord_uv_proj.copy()
    kp_coord_xy_proj[:, :2] = kp_coord_uv_proj[:, :2]*kp_coord_uv_proj[:, 2:]
    kp_coord_xy_proj = np.matmul(kp_coord_xy_proj,
                                 np.transpose(np.linalg.inv(K)))
    return kp_coord_xy_proj

# ========================================================
# EVALUATION
# ========================================================

def mean_L2_error(true, pred):
    """
    Calculate mean L2 error
    Args:
        true: GT keypoints (-1, 21, 3)
        pred: Predicted keypoints (-1, 21, 3)
    Out:
        Mean L2 error (-1, 1)
    """
    return np.mean(np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8))

def mean_L2_error_vis(true, pred, vis_list):
    """
    Calculate mean L2 error with visibility index
    Args:
        true        : GT keypoints (-1, 21, 3)
        pred        : Predicted keypoints (-1, 21, 3)
        vis_list    : Visibility index (21)
    Out:
        Mean L2 error (-1, 1)
    """
    error = []
    for i in range(len(pred)):
        cur_vis = vis_list[i]
        if len(pred[i][cur_vis]) == 0 or len(true[i][cur_vis]) == 0:
            error.append(0.0)
            continue
        error.append(mean_L2_error(true[i][cur_vis], pred[i][cur_vis]))
    return error, np.mean(error)

def calc_auc(y, x):
    """
    Calculate area under curve
    Args:
        y: error values
        x: thresholds
    Out:
        Area under curve
    """
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)
    return integral/norm

def percentage_frames_within_error_curve_vis(true, pred, vis_list, max_x=85,
                                         steps=5, plot=True):
    """
    Caclulate percentage of keypoints within a given error threshold and plot
    with visibility index
    Args:
        true            : GT points (-1, 21, 3)
        pred            : pred points (-1, 21, 3)
        vis_list        : Visibility index (21)
    Out:
        pck_curve_all   : List of PCK curve values
    """
    data_21_pts = np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8)
    data        = []
    for i in range(21):
        data.append([])

    for i in range(len(data_21_pts)):
        cur_vis = vis_list[i]
        for j in range(21):
            if cur_vis[j]:
                data[j].append(data_21_pts[i, j])
    data = np.asarray(data)

    error_threshold = np.arange(0, max_x, steps)
    pck_curve_all   = []
    for p_id in range(21):
        pck_curve = []
        for t in error_threshold:
            data_mean = np.mean((data[p_id] <= t).astype('float32'))
            pck_curve.append(data_mean)
        pck_curve_all.append(pck_curve)
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(error_threshold,pck_curve_all)
        ax.set_xticks(np.arange(0, max_x, steps))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        plt.grid()
        ax.set_ylabel('Frames within threshold %')
        ax.set_xlabel('Error Threshold mm')
        plt.show()
    return pck_curve_all

def percentage_frames_within_error_curve(true, pred, max_x=85,
                                         steps=5, plot=True):
    """
    Caclulate percentage of keypoints within a given error threshold and plot
    Args:
        true            : GT points (-1, 21, 3)
        pred            : pred points (-1, 21, 3)
    Out:
        pck_curve_all   : List of PCK curve values
    """
    data_21_pts     = np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8 )
    error_threshold = np.arange(0, max_x, steps)

    pck_curve_all = []
    for p_id in range(21):
        pck_curve = []
        for t in error_threshold:
            data_mean = np.mean((data_21_pts[:, p_id] <= t).astype('float32'))
            pck_curve.append(data_mean)
        pck_curve_all.append(pck_curve)
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(error_threshold,pck_curve_all)
        ax.set_xticks(np.arange(0, max_x, steps))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        plt.grid()
        ax.set_ylabel('Frames within threshold %')
        ax.set_xlabel('Error Threshold mm')
        plt.show()
    return pck_curve_all

def get_pck(data_21_pts, max_x=85, steps=5):
    """
    Caclulate percentage of keypoints within a given error threshold
    Args:
        data_21_pts     : List of 21 euclidean error values
                          corresponding to 1 hand
    Out:
        pck_curve_all   : List of PCK curve values
    """
    error_threshold = np.arange(0, max_x, steps)

    pck_curve_all = []
    for p_id in range(21):
        pck_curve = []
        for t in error_threshold:
            data_mean = np.mean((data_21_pts[:, p_id] <= t).astype('float32'))
            pck_curve.append(data_mean)
        pck_curve_all.append(pck_curve)
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)
    return pck_curve_all

def get_pck_with_vis(data, max_x=85, steps=5):
    """
    Caclulate percentage of keypoints within a given error threshold, used
    when accounting for visibility
    Args:
        data_21_pts     : List of 21 lists containing error values for each 
                          joint
    Out:
        pck_curve_all   : List of PCK curve values
    """
    error_threshold = np.arange(0, max_x, steps)

    pck_curve_all = []
    for p_id in range(21):
        pck_curve = []
        for t in error_threshold:
            data_mean = np.mean((data[p_id] <= t).astype('float32'))
            pck_curve.append(data_mean)
        pck_curve_all.append(pck_curve)
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)
    return pck_curve_all

def get_median_mean(true, pred, vis_list):
    """
    Caclulate median and mean similar to original code
    Args:
        true            : GT points (-1, 21, 3)
        pred            : pred points (-1, 21, 3)
        vis_list        : Visibility index (21)
    Out:
        median          : median
        mean            : mean
    """
    # https://github.com/lmb-freiburg/hand3d/blob/master/utils/general.py
    
    data_21_pts = np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8)
    data        = []
    for i in range(21):
        data.append([])

    for i in range(len(data_21_pts)):
        cur_vis = vis_list[i]
        for j in range(21):
            if cur_vis[j]:
                data[j].append(data_21_pts[i, j])
    data = np.asarray(data)
        
    median = []
    mean = []
    for p_id in range(21):
        pck_curve = []
        median.append(np.median(data[p_id]))
        mean.append(np.mean(data[p_id]))
    return np.mean(np.asarray(mean)), np.median(np.asarray(median))

# ========================================================
# UTILITIES
# ========================================================

def softmax(x):
    """ Compute softmax values for each sets of scores in x """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_hand_from_pred(pred_mask, filter_size=21):
    """
    Algorithm for obtaining hand from HandSegNet mask output
    Args:
        pred_mask: W, H, 2 output from HandSegNet
    Out:
        mask: hand mask for one mask
        bbox: corresponding bbox (x, y, w, h)
    """
    #https://github.com/lmb-freiburg/hand3d/blob/master/utils/general.py
    mask                            = pred_mask[:, :, 1].copy()
    ori_mask                        = mask.copy()
    ori_mask[ori_mask > 0]          = 1
    ori_mask[ori_mask <= 0]         = 0
    mask                            = softmax(mask)
    # Binary mask
    mask[mask == np.max(mask)]      = 1
    mask[~(mask == np.max(mask))]   = 0

    num_passes = mask.shape[0]//(filter_size//2)
    for i in range(num_passes):
        dilate = ndimage.grey_dilation(mask, size=(filter_size, filter_size))
        mask = ori_mask*dilate

    # Bbox
    y, x = np.where(mask == 1)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    w = x_max - x_min
    h = y_max - y_min
    x_cen = x_min + w/2
    y_cen = y_min + h/2

    box = np.asarray([x_cen, y_cen, w, h])

    return box, mask