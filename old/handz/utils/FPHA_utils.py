import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

from utils.dir import dir_dict

# ========================================================
# CONSTANTS
# ========================================================

DIR = dir_dict["FPHA_DIR"]

ORI_WIDTH = 1920
ORI_HEIGHT = 1080
REF_DEPTH = 1000

REORDER_IDX = [0, 1, 6, 7, 8, 2, 9,
               10, 11, 3, 12, 13, 14, 4,
               15, 16, 17, 5, 18, 19, 20]

CAM_EXTR = [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
            [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
            [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
            [0, 0, 0, 1]]

INV_CAM_EXTR = [[0.99998855624950122256, 0.0046911597684540387191, -0.00096970967236367877683, -25.701645303388132272],
                [-0.0046884842637616731197, 0.99998527559956268165,
                 0.0027430368219501163773, -1.1101913203320408265],
                [0.00098256339938933108913, -0.0027384588555197885184, 0.99999576732453258074, -3.9238944436608977969]]

FOCAL_LENGTH_X_COLOR = 1395.749023
FOCAL_LENGTH_Y_COLOR = 1395.749268
X0_COLOR = 935.732544
Y0_COLOR = 540.681030

CAM_INTR_COLOR = [[FOCAL_LENGTH_X_COLOR, 0, X0_COLOR],
                  [0, FOCAL_LENGTH_Y_COLOR, Y0_COLOR],
                  [0, 0, 1]]

FOCAL_LENGTH_X_DEPTH = 475.065948
FOCAL_LENGTH_Y_DEPTH = 475.065857
X0_DEPTH = 315.944855
Y0_DEPTH = 245.287079

CAM_INTR_DEPTH = [[FOCAL_LENGTH_X_DEPTH, 0, X0_DEPTH],
                  [0, FOCAL_LENGTH_Y_DEPTH, Y0_DEPTH],
                  [0, 0, 1]]

# ========================================================
# VISUALISATION
# ========================================================

def visualize_joints_2d(ax, joints, joint_idxs=False, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
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
    _draw2djoints(ax, joints, links, alpha=alpha)

def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
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
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)

def visualize_joints_3d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]
    ax.scatter(x, y, z, c="red")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw3djoints(ax, joints, links, alpha=alpha)

def _draw3djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ["r", "m", "b", "c", "g"]
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw3dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)

def _draw3dseg(ax, annot, idx1, idx2, c="r", alpha=1):
    """Draw segment of given color"""

    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]], [annot[idx1, 2], annot[idx2, 2]],
        c=c,
        alpha=alpha)

def draw_bbox(ax, box, img_dim, c='r'):
    """
    boxes: (x_center, y_center, width, height)
    img_dim: (W, H)
    output: plot boxes on img
    """

    x1 = (box[0] - box[2]/2.0)*img_dim[0]
    y1 = (box[1] - box[3]/2.0)*img_dim[1]
    width = box[2]*img_dim[0]
    height = box[3]*img_dim[1]        
    
    import matplotlib.patches as patches
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=c, facecolor='none')
    ax.add_patch(rect)

# ========================================================
# MISC
# ========================================================

def is_root_in_img(uvd_gt, root_id=3):
    """
    Check if root is present in image
    """
    if uvd_gt[root_id, 0] > ORI_WIDTH or uvd_gt[root_id, 0] < 0 or \
    uvd_gt[root_id, 1] > ORI_HEIGHT or uvd_gt[root_id, 1] < 0:
        return False
    else:
        return True 

def is_hand_in_img(img, uvd_gt, pad=50):
    """
    Check if hand is present in image
    """
    x_max = int(np.amax(uvd_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt[:,1])) - pad, 0)
    crop_hand = img[y_min:y_max, x_min:x_max, :]
    if crop_hand.size == 0:
        return False
    else:
        return True

def is_annot_outside_img(uvd_gt, img_dim_WH=(1920, 1080)):
    """
    check if ground truth has point outside image
    """
    for pt in uvd_gt:
        if pt[0] < 0 or pt[0] > img_dim_WH[0] \
            or pt[1] < 0 or pt[1] > img_dim_WH[1]:
                return True
    return False

# ========================================================
# GET GROUND TRUTH
# ========================================================

def get_bbox(uvd_gt, img_size=(ORI_WIDTH, ORI_HEIGHT), pad=50):
    """
    get normalized bounding box for hand
    input: 21,3 hand points in img space
    output: x,y (center), w, h
    """
    x_max = int(np.amax(uvd_gt[:,0])) + pad
    x_min = np.maximum(int(np.amin(uvd_gt[:,0])) - pad, 0)
    y_max = int(np.amax(uvd_gt[:,1])) + pad
    y_min = np.maximum(int(np.amin(uvd_gt[:,1])) - pad, 0)

    # ensure bbox is within img bounds
    if y_max > ORI_HEIGHT:
        y_max = ORI_HEIGHT
    if y_min < 0:
        y_min = 0
    if x_max > ORI_WIDTH:
        x_max = ORI_WIDTH
    if x_min < 0:
        x_min = 0

    width = (x_max - x_min) / img_size[0]
    height = (y_max - y_min) / img_size[1]
    x_cen = ((x_min + x_max)/2) / img_size[0]
    y_cen = ((y_min + y_max)/2) / img_size[1]
    
    return np.asarray([x_cen, y_cen, width, height])

def get_train_test_pairs(modality, dataset_dir):
    # print("READING DATA FROM: data_split_action_recognition.txt")
    img_dir = os.path.join(dataset_dir, "Video_files")
    skel_dir = os.path.join(dataset_dir, "Hand_pose_annotation_v1")
    if modality == "depth":
        img_type = "png"
    else:
        img_type = "jpeg"
    train_file_name = []
    test_file_name = []
    train_xyz_gt = []
    test_xyz_gt = []
    with open(os.path.join(dataset_dir, "data_split_action_recognition.txt")) as f:
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
                skeleton_vals = np.loadtxt(skeleton_path)
                for i in range(len_frame_idx):
                    img_path = os.path.join(path, modality, 
                                            "%s_%04d.%s" %(modality, i, img_type))
                    skel_xyz = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, 3)[i]
                    if cur_split == "Training":
                        train_file_name.append(img_path)
                        train_xyz_gt.append(skel_xyz)
                    else:
                        test_file_name.append(img_path)
                        test_xyz_gt.append(skel_xyz)
    
    return train_file_name, test_file_name, np.asarray(train_xyz_gt), np.asarray(test_xyz_gt)

def get_video_pairs(modality, dataset_dir, subject, action_name, seq_idx):
    # print("READING DATA FROM: data_split_action_recognition.txt")
    img_dir = os.path.join(dataset_dir, "Video_files")
    skel_dir = os.path.join(dataset_dir, "Hand_pose_annotation_v1")
    if modality == "depth":
        img_type = "png"
    else:
        img_type = "jpeg"    
    file_name = []
    xyz_gt = []        
    path = os.path.join(subject, action_name, str(seq_idx))
    full_path = os.path.join(img_dir, path, modality)
    len_frame_idx = len([x for x in os.listdir(full_path)
                        if os.path.join(full_path, x)])
    skeleton_path = os.path.join(skel_dir, path, "skeleton.txt")
    skeleton_vals = np.loadtxt(skeleton_path)
    for i in tqdm(range(len_frame_idx)):
        img_path = os.path.join(path, modality, 
                                "%s_%04d.%s" %(modality, i, img_type))
        skel_xyz = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, 3)[i]
        file_name.append(img_path)
        xyz_gt.append(skel_xyz)

    return file_name, np.asarray(xyz_gt)

def get_GT_pairs(modality, dataset_dir, split):
    # print(f"GETTING DATA FOR SPLIT: {split}")
    if split == "train":
        file_name, _, xyz_gt, _ = get_train_test_pairs(modality, dataset_dir)
    elif split == "test":
        _, file_name, _, xyz_gt = get_train_test_pairs(modality, dataset_dir)
    elif split == "val":
        _, file_name, _, xyz_gt = get_train_test_pairs(modality, dataset_dir)   
    else:
        raise ValueError(f"Invalid split: {split}")     
    return file_name, xyz_gt
    
def load_objects(obj_root):
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

def get_img(frame, video_file = "Video_files"):
    img = Image.open(os.path.join(DIR, video_file, frame))
    img = np.asarray(img, dtype="uint32")
    return img

def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample["subject"],
                                 sample["action_name"], sample["seq_idx"],
                                 "skeleton.txt")
    # print("Loading skeleton from {}".format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                            -1)[sample["frame_idx"]]
    return skeleton

def get_obj_transform(sample, obj_root):
    seq_path = os.path.join(obj_root, sample["subject"], sample["action_name"],
                            sample["seq_idx"], "object_pose.txt")
    with open(seq_path, "r") as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[sample["frame_idx"]]
    line = raw_line.strip().split(" ")
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    # print("Loading obj transform from {}".format(seq_path))
    return trans_matrix

# ========================================================
# XYZ UVD CONVERSION
# ========================================================

def uvd2xyz_depth(skel_uvd):
    skel_xyz = np.empty_like(skel_uvd).astype("float32")
    skel_xyz[...,0] = (skel_uvd[..., 0]-X0_DEPTH)/FOCAL_LENGTH_X_DEPTH*skel_uvd[..., 2]
    skel_xyz[...,1] = (skel_uvd[..., 1]-Y0_DEPTH)/FOCAL_LENGTH_Y_DEPTH*skel_uvd[..., 2]
    skel_xyz[...,2] = skel_uvd[...,2]
    return skel_xyz

def xyz2uvd_depth(skel_xyz):
    skel_uvd = np.empty_like(skel_xyz).astype("float32")
    skel_uvd[..., 0] = X0_DEPTH+FOCAL_LENGTH_X_DEPTH*(skel_xyz[..., 0]/skel_xyz[..., 2])
    skel_uvd[..., 1] = Y0_DEPTH+FOCAL_LENGTH_Y_DEPTH*(skel_xyz[..., 1]/skel_xyz[..., 2])
    skel_uvd[..., 2] = skel_xyz[..., 2]
    return skel_uvd

def xyz2uvd_color(skel_xyz):
    skel_uvd = np.empty_like(skel_xyz).astype("float32")
    ccs_x = CAM_EXTR[0][0]*skel_xyz[..., 0]+CAM_EXTR[0][1]*skel_xyz[..., 1]+CAM_EXTR[0][2]*skel_xyz[..., 2]+CAM_EXTR[0][3]
    ccs_y = CAM_EXTR[1][0]*skel_xyz[..., 0]+CAM_EXTR[1][1]*skel_xyz[..., 1]+CAM_EXTR[1][2]*skel_xyz[..., 2]+CAM_EXTR[1][3]
    ccs_z = CAM_EXTR[2][0]*skel_xyz[..., 0]+CAM_EXTR[2][1]*skel_xyz[..., 1]+CAM_EXTR[2][2]*skel_xyz[..., 2]+CAM_EXTR[2][3]

    skel_uvd[..., 0] = X0_COLOR+FOCAL_LENGTH_X_COLOR*(ccs_x/ccs_z)
    skel_uvd[..., 1] = Y0_COLOR+FOCAL_LENGTH_Y_COLOR*(ccs_y/ccs_z)
    skel_uvd[..., 2] = ccs_z
    return skel_uvd

def uvd2xyz_color(skel_uvd):
    ccs_z = skel_uvd[..., 2]
    ccs_x = ((skel_uvd[..., 0]-X0_COLOR)/FOCAL_LENGTH_X_COLOR)*ccs_z
    ccs_y = ((skel_uvd[..., 1]-Y0_COLOR)/FOCAL_LENGTH_Y_COLOR)*ccs_z

    skel_xyz = np.empty_like(skel_uvd).astype("float32")
    skel_xyz[..., 0] = INV_CAM_EXTR[0][0]*ccs_x+INV_CAM_EXTR[0][1]*ccs_y+INV_CAM_EXTR[0][2]*ccs_z+INV_CAM_EXTR[0][3]
    skel_xyz[..., 1] = INV_CAM_EXTR[1][0]*ccs_x+INV_CAM_EXTR[1][1]*ccs_y+INV_CAM_EXTR[1][2]*ccs_z+INV_CAM_EXTR[1][3]
    skel_xyz[..., 2] = INV_CAM_EXTR[2][0]*ccs_x+INV_CAM_EXTR[2][1]*ccs_y+INV_CAM_EXTR[2][2]*ccs_z+INV_CAM_EXTR[2][3]
    return skel_xyz

def xyz2ccs_color(skel_xyz):
    skel_ccs = np.empty_like(skel_xyz).astype("float32")
    skel_ccs[..., 0] = CAM_EXTR[0][0]*skel_xyz[..., 0]+CAM_EXTR[0][1]*skel_xyz[..., 1]+CAM_EXTR[0][2]*skel_xyz[..., 2]+CAM_EXTR[0][3]
    skel_ccs[..., 1] = CAM_EXTR[1][0]*skel_xyz[..., 0]+CAM_EXTR[1][1]*skel_xyz[..., 1]+CAM_EXTR[1][2]*skel_xyz[..., 2]+CAM_EXTR[1][3]
    skel_ccs[..., 2] = CAM_EXTR[2][0]*skel_xyz[..., 0]+CAM_EXTR[2][1]*skel_xyz[..., 1]+CAM_EXTR[2][2]*skel_xyz[..., 2]+CAM_EXTR[2][3]
    return skel_ccs

def ccs2uvd_color(skel_ccs):
    skel_uvd = np.empty_like(skel_ccs).astype("float32")
    skel_uvd[..., 0] = X0_COLOR+FOCAL_LENGTH_X_COLOR*(skel_ccs[..., 0]/skel_ccs[..., 2])
    skel_uvd[..., 1] = Y0_COLOR+FOCAL_LENGTH_Y_COLOR*(skel_ccs[..., 1]/skel_ccs[..., 2])
    skel_uvd[..., 2] = skel_ccs[..., 2]
    return skel_uvd

def uvd2ccs_color(skel_uvd):
    skel_ccs = np.empty_like(skel_uvd).astype("float32")
    skel_ccs[..., 2] = skel_uvd[..., 2]
    skel_ccs[..., 0] = ((skel_uvd[..., 0] - X0_COLOR)/FOCAL_LENGTH_X_COLOR)*skel_uvd[..., 2]
    skel_ccs[..., 1] = ((skel_uvd[..., 1]- Y0_COLOR)/FOCAL_LENGTH_Y_COLOR)*skel_uvd[..., 2]
    return skel_ccs

def ccs2xyz_color(skel_ccs):
    skel_xyz = np.empty_like(skel_ccs).astype("float32")
    skel_xyz[..., 0] = INV_CAM_EXTR[0][0]*skel_ccs[..., 0]+INV_CAM_EXTR[0][1]*skel_ccs[..., 1]+INV_CAM_EXTR[0][2]*skel_ccs[..., 2]+INV_CAM_EXTR[0][3]
    skel_xyz[..., 1] = INV_CAM_EXTR[1][0]*skel_ccs[..., 0]+INV_CAM_EXTR[1][1]*skel_ccs[..., 1]+INV_CAM_EXTR[1][2]*skel_ccs[..., 2]+INV_CAM_EXTR[1][3]
    skel_xyz[..., 2] = INV_CAM_EXTR[2][0]*skel_ccs[..., 0]+INV_CAM_EXTR[2][1]*skel_ccs[..., 1]+INV_CAM_EXTR[2][2]*skel_ccs[..., 2]+INV_CAM_EXTR[2][3]
    return skel_xyz
