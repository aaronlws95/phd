from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np

REORDER = [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]

def show_img_and_skel_color(file_name, uvd_gt):
    if len(uvd_gt.shape) == 1:
        uvd_gt = np.reshape(uvd_gt, (21, 3))

    color = Image.open(file_name)
    color = np.asarray(color, dtype='uint32')
    fig, ax = plt.subplots()
    ax.imshow(color)
    visualize_joints_2d(ax, uvd_gt[REORDER], joint_idxs=False)
    plt.show()

def show_true_pred_img_and_skel_color(file_name, uvd_gt, pred_uvd):
    if len(uvd_gt.shape) == 1:
        uvd_gt = np.reshape(uvd_gt, (21, 3))
    if len(pred_uvd.shape) == 1:
        pred_uvd = np.reshape(pred_uvd, (21, 3))

    color = Image.open(file_name)
    color = np.asarray(color, dtype='uint32')
    fig, ax = plt.subplots(1,2, figsize=(18, 10))
    ax[0].imshow(color)
    ax[1].imshow(color)
    ax[0].set_title('pred')
    ax[1].set_title('true')
    visualize_joints_2d(ax[0], pred_uvd[REORDER], joint_idxs=False)
    visualize_joints_2d(ax[1], uvd_gt[REORDER], joint_idxs=False)
    plt.show()

def show_3d_true_pred_skel(file_name, uvd_gt, pred_uvd):
    if len(uvd_gt.shape) == 1:
        uvd_gt = np.reshape(uvd_gt, (21, 3))
    if len(pred_uvd.shape) == 1:
        pred_uvd = np.reshape(pred_uvd, (21, 3))

    fig = plt.figure(figsize=(12, 5))
    ax_1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax_1.set_title('pred')
    ax_2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax_2.set_title('true')

    for ax in [ax_1, ax_2]:
        ax.view_init(elev=30, azim=45)

    visualize_joints_3d(ax_1, pred_uvd[REORDER], joint_idxs=False)
    visualize_joints_3d(ax_2, uvd_gt[REORDER], joint_idxs=False)
    plt.show()

def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)

def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)

def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
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
    ax.scatter(x, y, z, c='red')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw3djoints(ax, joints, links, alpha=alpha)

def _draw3djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw3dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)

def _draw3dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""

    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]], [annot[idx1, 2], annot[idx2, 2]],
        c=c,
        alpha=alpha)
