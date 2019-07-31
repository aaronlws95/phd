import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def draw_joints(ax, joints, c=None):
    """
    Draw 2d skeleton on matplotlib axis
    Args:
        joints : Hand keypoints (21, 2++)
    """
    def _draw2djoints(ax, joints, links, c=None):
        """ Draw segments """
        if c:
            colors = [c, c, c, c, c]
        else:
            colors = ['r', 'm', 'b', 'c', 'g']
        for finger_idx, finger_links in enumerate(links):
            for idx in range(len(finger_links) - 1):
                _draw2dseg(
                    ax,
                    joints,
                    finger_links[idx],
                    finger_links[idx + 1],
                    c=colors[finger_idx])
                
    def _draw2dseg(ax, annot, idx1, idx2, c=None):
        """ Draw segment of given color """
        ax.plot([annot[idx1, 0], annot[idx2, 0]], 
                [annot[idx1, 1], annot[idx2, 1]], c=c)
    
    links = [(0, 1, 6, 7, 8), (0, 2, 9, 10, 11), (0, 3, 12, 13, 14),
             (0, 4, 15, 16, 17), (0, 5, 18, 19, 20)]
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 2, 'r')
    _draw2djoints(ax, joints, links, c=c)

def draw_bbox(ax, box, c='r'):
    """
    Draw bounding box
    Args:
        boxes   : [x_cen, y_cen, w, h]
        img_dim : (W, H)
        c       : Fix color of bounding box
    """
    x1      = (box[0] - box[2]/2.0)
    y1      = (box[1] - box[3]/2.0)
    width   = box[2]
    height  = box[3]
    rect    = patches.Rectangle((x1, y1), width, height,
                                linewidth=2, edgecolor=c, facecolor='none')
    ax.add_patch(rect)

def show_spatial_temporal_seq(img_list, num_segments, spatial_mask, attn_weight,
                              spatial_thr=0.5, figsize=(20, 20)):
    fig, ax = plt.subplots(1, num_segments, figsize=figsize)
    for i in range(len(img_list)):
        ax[i].imshow(img_list[i])
    plt.show()
    
    fig, ax = plt.subplots(1, num_segments, figsize=figsize)
    for i in range(len(img_list)):
        spatial_overlay = cv2.resize(spatial_mask[i], (img_list[i].shape[1], img_list[i].shape[0]))
        spatial_overlay = spatial_overlay > spatial_thr
        ax[i].imshow(spatial_overlay, cmap='jet')
        ax[i].imshow(img_list[i], alpha=0.5)
    plt.show()
    
    fig, ax = plt.subplots(1, num_segments, figsize=figsize)
    temporal_attn = np.argsort(attn_weight)
    for i in range(len(img_list)):
        temp_id = np.where(temporal_attn == i)[0][0] + 1
        temp_overlay = np.ones(img_list[i].shape)*(1/temp_id)
        ax[i].imshow(temp_overlay)
        ax[i].text(img_list[i].shape[1]//2, img_list[i].shape[0]//2, 
                    str(temp_id), 
                    dict(size=30),
                    color='r')
        ax[i].imshow(img_list[i], alpha=0.5)
    plt.show()