import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import mpl_toolkits.mplot3d as plt3d

import cv2

def draw_3D_joints(ax, joints, c=None):
    """ draw 3d skeleton on matplotlib axis """
    def _draw3djoints(ax, joints, links, c=None):
        """ draw segments """
        if c:
            colors = [c, c, c, c, c]
        else:
            colors = ['r', 'm', 'b', 'c', 'g']
        for finger_idx, finger_links in enumerate(links):
            for idx in range(len(finger_links) - 1):
                _draw3dseg(
                    ax,
                    joints,
                    finger_links[idx],
                    finger_links[idx + 1],
                    c=colors[finger_idx])
    def _draw3dseg(ax, annot, idx1, idx2, c=None):
        """ draw segment of given color """
        ax.plot(
            [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]], [annot[idx1, 2], annot[idx2, 2]],
            c=c)

    links = [(0, 1, 6, 7, 8), (0, 2, 9, 10, 11), (0, 3, 12, 13, 14),
             (0, 4, 15, 16, 17), (0, 5, 18, 19, 20)]
    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]
    ax.scatter(x, y, z, c=c)
    _draw3djoints(ax, joints, links, c=c)
    
def draw_joints(ax, joints, c=None):
    """ draw 2d skeleton on matplotlib axis """
    def _draw2djoints(ax, joints, links, c=None):
        """ draw segments """
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
        """ draw segment of given color """
        ax.plot([annot[idx1, 0],
        annot[idx2, 0]],
                [annot[idx1, 1], annot[idx2, 1]], c=c)

    links = [(0, 1, 6, 7, 8), (0, 2, 9, 10, 11), (0, 3, 12, 13, 14),
             (0, 4, 15, 16, 17), (0, 5, 18, 19, 20)]
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 2, 'r')
    _draw2djoints(ax, joints, links, c=c)

def draw_bbox(ax, box, mode='xywh', c='r', linewidth=2):
    """ draw bounding box """
    if mode == 'xywh':
        x1      = (box[0] - box[2]/2.0)
        y1      = (box[1] - box[3]/2.0)
        width   = box[2]
        height  = box[3]
    elif mode == 'xyxy':
        x1 = box[0]
        y1 = box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
    else:
        raise ValueError('Invalid mode', mode)

    rect = patches.Rectangle((x1, y1), width, height, linewidth=linewidth, edgecolor=c, facecolor='none')
    ax.add_patch(rect)

def draw_obj_joints(ax, joints, linewidth=2, c='g'):

    links = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]

    for j in joints:
        ax.scatter(j[0], j[1], 30, c)

    # draws lines connected using links
    for i in range(len(links)):
        for j in range(len(links[i]) - 1):
            jntC = links[i][j]
            jntN = links[i][j+1]
            l = mlines.Line2D([joints[jntC,0], joints[jntN,0]], [joints[jntC,1], joints[jntN,1]], color=c, linewidth=linewidth)
            ax.add_line(l)

def draw_obj_3D_joints(ax, joints, linewidth=2, c='g'):

    links = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]

    for j in joints:
        ax.scatter(j[0], j[1], j[2], c=c)

    # draws lines connected using links
    for i in range(len(links)):
        for j in range(len(links[i]) - 1):
            jntC = links[i][j]
            jntN = links[i][j+1]
            l = plt3d.art3d.Line3D([joints[jntC,0], joints[jntN,0]], [joints[jntC,1], joints[jntN,1]], [joints[jntC,2], joints[jntN,2]], color=c, linewidth=linewidth)
            ax.add_line(l)
