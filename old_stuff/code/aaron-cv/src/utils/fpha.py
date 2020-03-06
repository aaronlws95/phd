import os
import cv2
import numpy                                as np
import matplotlib.patches                   as patches
from tqdm                   import tqdm
from PIL                    import Image
from matplotlib             import pyplot   as plt
from mpl_toolkits.mplot3d   import Axes3D

from src.utils              import IMG, DATA_DIR

# ========================================================
# CONSTANTS
# ========================================================

ORI_WIDTH               = 1920
ORI_HEIGHT              = 1080
REF_DEPTH               = 1000

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
# VISUALISATION
# ========================================================

def visualize_paf(img, pafs, stride=8):
    colors = [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255],
              [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 0],
              [255, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 0, 255],
              [0, 0, 255], [0, 0, 255], [255, 0, 255], [255, 0, 255], [255, 0, 255]]    
    
    img = img.copy()
    for i in range(pafs.shape[2]//2):
        paf_x = pafs[:,:, i*2]
        paf_y = pafs[:,:, i*2+1]
        len_paf = np.sqrt(paf_x**2 + paf_y**2 + 1e-8)
        for x in range(0, img.shape[0], stride):
            for y in range(0, img.shape[1], stride):
                if len_paf[x,y]>0.25:
                    img = cv2.arrowedLine(img, (y,x), (int(y + 10*paf_x[x,y]), int(x + 10*paf_y[x,y])), colors[i], 1)
    return img

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

def get_bbox(uvd_gt, img_size=(ORI_WIDTH, ORI_HEIGHT), pad=50):
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

def get_train_test_pairs_with_action(modality, dataset_dir):
    """ Get training and test pairs for FPHA
    Args:
        modality        : [color, depth]
    Out:
        train_file_name : training file names
        test_file_name  : testing file names
        train_xyz_gt    : training XYZ annotations
        test_xyz_gt     : testing XYZ annotations
    """
    if modality == "depth":
        img_type = "png"
    else:
        img_type = "jpeg"

    img_dir         = os.path.join(dataset_dir, "Video_files")
    skel_dir        = os.path.join(dataset_dir, "Hand_pose_annotation_v1")
    train_file_name = []
    test_file_name  = []
    train_xyz_gt    = []
    test_xyz_gt     = []
    train_action_id = []
    test_action_id = []
    with open(os.path.join(dataset_dir,
                           "data_split_action_recognition.txt")) as f:
        cur_split = "Training"
        lines = f.readlines()
        for l in tqdm(lines):
            words = l.split()
            if(words[0] == "Training" or words[0] == "Test"):
                cur_split = words[0]
            else:
                path = l.split()[0]
                action_id = l.split()[1]
                full_path = os.path.join(img_dir, path, modality)
                len_frame_idx = len([x for x in os.listdir(full_path)
                                    if os.path.join(full_path, x)])
                skeleton_path = os.path.join(skel_dir, path, "skeleton.txt")
                sk_val = np.loadtxt(skeleton_path)
                for i in range(len_frame_idx):
                    img_path = os.path.join(path, modality,
                                            "%s_%04d.%s" %(modality, i,
                                                           img_type))
                    skel_xyz = sk_val[:, 1:].reshape(sk_val.shape[0], 21, 3)[i]
                    if cur_split == "Training":
                        train_file_name.append(img_path)
                        train_xyz_gt.append(skel_xyz)
                        train_action_id.append(action_id)
                    else:
                        test_file_name.append(img_path)
                        test_xyz_gt.append(skel_xyz)
                        test_action_id.append(action_id)

    return train_file_name, test_file_name, \
            np.asarray(train_xyz_gt), np.asarray(test_xyz_gt), \
            train_action_id, test_action_id

def get_train_test_pairs(modality, dataset_dir):
    """ Get training and test pairs for FPHA
    Args:
        modality        : [color, depth]
    Out:
        train_file_name : training file names
        test_file_name  : testing file names
        train_xyz_gt    : training XYZ annotations
        test_xyz_gt     : testing XYZ annotations
    """
    if modality == "depth":
        img_type = "png"
    else:
        img_type = "jpeg"

    img_dir         = os.path.join(dataset_dir, "Video_files")
    skel_dir        = os.path.join(dataset_dir, "Hand_pose_annotation_v1")
    train_file_name = []
    test_file_name  = []
    train_xyz_gt    = []
    test_xyz_gt     = []

    with open(os.path.join(dataset_dir,
                           "data_split_action_recognition.txt")) as f:
        cur_split = "Training"
        lines = f.readlines()
        for l in tqdm(lines):
            words = l.split()
            if(words[0] == "Training" or words[0] == "Test"):
                cur_split = words[0]
            else:
                path = l.split()[0]
                full_path = os.path.join(img_dir, path, modality)
                len_frame_idx = len([x for x in os.listdir(full_path)
                                    if os.path.join(full_path, x)])
                skeleton_path = os.path.join(skel_dir, path, "skeleton.txt")
                sk_val = np.loadtxt(skeleton_path)
                for i in range(len_frame_idx):
                    img_path = os.path.join(path, modality,
                                            "%s_%04d.%s" %(modality, i,
                                                           img_type))
                    skel_xyz = sk_val[:, 1:].reshape(sk_val.shape[0], 21, 3)[i]
                    if cur_split == "Training":
                        train_file_name.append(img_path)
                        train_xyz_gt.append(skel_xyz)
                    else:
                        test_file_name.append(img_path)
                        test_xyz_gt.append(skel_xyz)

    return train_file_name, test_file_name, \
            np.asarray(train_xyz_gt), np.asarray(test_xyz_gt)

def crop_hand(img, uvd_gt, img_size=(ORI_WIDTH, ORI_HEIGHT), pad=10):
    """
    Get cropped image and cropped hand from image
    Args:
        img         : (ORI_WIDTH, ORI_HEIGHT, 3)
        uvd_gt      : Hand points in img space (21, 2++)
    Out:
        crop        : Cropped image
        uvd_gt_crop : [x_cen, y_cen, w, h]
        x_min       : X value for translation
        y_min       : Y value for translation
    """
    uvd_gt_rsz = IMG.scale_points_WH(uvd_gt,
                                     (ORI_WIDTH, ORI_HEIGHT),
                                     (img.shape[1], img.shape[0]))

    x_max = int(np.amax(uvd_gt_rsz[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt_rsz[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_gt_rsz[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt_rsz[:,1])) - pad, 0)

    # Ensure bbox is within img bounds
    if y_max > img.shape[0]:
        y_max = img.shape[0]
    if y_min < 0:
        y_min = 0
    if x_max > img.shape[1]:
        x_max = img.shape[1]
    if x_min < 0:
        x_min = 0

    crop = img[y_min:y_max, x_min:x_max, :]

    uvd_gt_crop = uvd_gt_rsz
    uvd_gt_crop[:, 0] = uvd_gt_crop[:, 0] - x_min
    uvd_gt_crop[:, 1] = uvd_gt_crop[:, 1] - y_min
    return crop, uvd_gt_crop, x_min, y_min

def crop_hand_from_bbox(img, bbox):
    """
    Get cropped hand given bounding box
    Args:
        img     : (H, W, C)
        bbox    : [x_cen, y_cen, w, h]
    Out:
        crop    : Cropped image
    """
    x_min = int(bbox[0] - bbox[2]/2)
    y_min = int(bbox[1] - bbox[3]/2)
    x_max = int(bbox[0] + bbox[2]/2)
    y_max = int(bbox[1] + bbox[3]/2)

    if y_max > img.shape[0]:
        y_max = img.shape[0]
    if y_min < 0:
        y_min = 0
    if x_max > img.shape[1]:
        x_max = img.shape[1]
    if x_min < 0:
        x_min = 0

    crop = img[y_min:y_max, x_min:x_max, :]
    return crop

def get_video_pairs(modality, dataset_dir, subject, action_name, seq_idx):
    """ Get pairs given FPHA sequence info """
    if modality == "depth":
        img_type = "png"
    else:
        img_type = "jpeg"

    img_dir         = os.path.join(dataset_dir, "Video_files")
    skel_dir        = os.path.join(dataset_dir, "Hand_pose_annotation_v1")
    file_name       = []
    xyz_gt          = []
    path            = os.path.join(subject, action_name, str(seq_idx))
    full_path       = os.path.join(img_dir, path, modality)
    len_frame_idx   = len([x for x in os.listdir(full_path)
                           if os.path.join(full_path, x)])
    skeleton_path   = os.path.join(skel_dir, path, "skeleton.txt")
    sk_val          = np.loadtxt(skeleton_path)

    for i in tqdm(range(len_frame_idx)):
        img_path = os.path.join(path, modality,
                                "%s_%04d.%s" %(modality, i, img_type))
        skel_xyz = sk_val[:, 1:].reshape(sk_val.shape[0], 21, 3)[i]
        file_name.append(img_path)
        xyz_gt.append(skel_xyz)
    return file_name, np.asarray(xyz_gt)

def get_skeleton(sample, skel_root):
    """ Get hand keypoints """
    skeleton_path   = os.path.join(skel_root, sample["subject"],
                                   sample["action_name"], sample["seq_idx"],
                                   "skeleton.txt")
    sk_val          = np.loadtxt(skeleton_path)
    skeleton        = sk_val[:, 1:].reshape(sk_val.shape[0], 21, -1)
    skeleton        = skeleton[sample["frame_idx"]]
    return skeleton

def load_objects(obj_root):
    """ Load objects """
    object_names = ["juice", "liquid_soap", "milk", "salt"]
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, "{}_model".format(obj_name),
                                "{}_model.ply".format(obj_name))
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = {
            "verts": np.array(mesh.vertices),
            "faces": np.array(mesh.faces)
        }
    return all_models

def get_obj_transform(sample, obj_root):
    """ Get object transformation """
    seq_path = os.path.join(obj_root, sample["subject"], sample["action_name"],
                            sample["seq_idx"], "object_pose.txt")
    with open(seq_path, "r") as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[sample["frame_idx"]]
    line = raw_line.strip().split(" ")
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    return trans_matrix

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
    return integral / norm

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
    # (b, 21) error values for each individual keypoint
    data_21_pts     = np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8)
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
        ax.plot(error_threshold, pck_curve_all)
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

# ========================================================
# NORMUVD
# ========================================================

def get_bbox_for_normuvd(bbsize=BBSIZE, ref_z=REF_DEPTH,
                         u0=X0_COLOR, v0=Y0_COLOR):
    """ Get BBOX for normuvd """
    bbox_xyz = np.array([(bbsize,bbsize,ref_z)])
    bbox_uvd = xyz2uvd_color(bbox_xyz)
    bbox_uvd[0,0] = np.ceil(bbox_uvd[0,0] - u0)
    bbox_uvd[0,1] = np.ceil(bbox_uvd[0,1] - v0)
    return bbox_xyz, bbox_uvd

def get_normuvd(xyz_gt, center_joint_idx=3):
    """ XYZ keypoints to normalized UVD form """
    # Mean
    xyz_hand_gt = xyz_gt.copy()
    mean_z = xyz_hand_gt[center_joint_idx, 2]
    xyz_hand_gt[:, 2] += REF_DEPTH - mean_z
    uvd_hand_gt = xyz2uvd_color(xyz_hand_gt)
    mean_u = uvd_hand_gt[center_joint_idx, 0]
    mean_v = uvd_hand_gt[center_joint_idx, 1]
    # U
    uvd_hand_gt[:, 0] = \
        (uvd_hand_gt[:, 0] - mean_u + BBOX_NORMUVD[0]/2)/BBOX_NORMUVD[0]
    uvd_hand_gt[np.where(uvd_hand_gt[:, 0] > 1), 0] = 1
    uvd_hand_gt[np.where(uvd_hand_gt[:, 0] < 0), 0] = 0
    #V
    uvd_hand_gt[:,1] = \
        (uvd_hand_gt[:,1] - mean_v + BBOX_NORMUVD[1]/2)/BBOX_NORMUVD[1]
    uvd_hand_gt[np.where(uvd_hand_gt[:, 1] > 1), 1] = 1
    uvd_hand_gt[np.where(uvd_hand_gt[:, 1] < 0), 1] = 0
    # Z
    uvd_hand_gt[:,2] = (uvd_hand_gt[:, 2] - REF_DEPTH + BBSIZE/2)/BBSIZE
    return uvd_hand_gt, (mean_u, mean_v, mean_z)

def normuvd2xyzuvd_color(norm_uvd, hand_center_uvd):
    """ Normalized UVD form to XYZ and UVD space """
    if len(norm_uvd.shape) == 3:
        mean_u = np.expand_dims(hand_center_uvd[..., 0], axis=-1)
        mean_v = np.expand_dims(hand_center_uvd[..., 1], axis=-1)
        mean_z = np.expand_dims(hand_center_uvd[..., 2], axis=-1)
    else:
        mean_u = hand_center_uvd[0]
        mean_v = hand_center_uvd[1]
        mean_z = hand_center_uvd[2]

    uvd_hand = np.empty_like(norm_uvd).astype('float32')
    uvd_hand[..., 0] = \
        norm_uvd[..., 0]*BBOX_NORMUVD[0] + mean_u - BBOX_NORMUVD[0]/2
    uvd_hand[..., 1] = \
        norm_uvd[..., 1]*BBOX_NORMUVD[1] + mean_v - BBOX_NORMUVD[1]/2
    uvd_hand[..., 2] = norm_uvd[..., 2]*BBSIZE + REF_DEPTH - BBSIZE/2
    
    xyz         = uvd2xyz_color(uvd_hand)
    xyz[... ,2] = xyz[... ,2] - REF_DEPTH + mean_z
    uvd         = xyz2uvd_color(xyz)
    return xyz, uvd

# ========================================================
# ACTION RECOGNITION
# ========================================================

def get_class_name_dict():
    cn_dict = {}
    action_list_dir = 'First_Person_Action_Benchmark/action_object_info.txt'
    with open(os.path.join(DATA_DIR, action_list_dir), 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                l = l.split(' ')
                cn_dict[int(l[0]) - 1] = l[1]
    return cn_dict

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