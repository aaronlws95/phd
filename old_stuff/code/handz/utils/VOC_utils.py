from PIL import Image
from matplotlib import pyplot as plt 
import os
import math 
import numpy as np

from utils.dir import dir_dict

DIR = dir_dict["VOC_DIR"]

SETS = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
CLASS_LIST = ["aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", "diningtable", 
            "dog", "horse", "motorbike", "person", "pottedplant", 
            "sheep", "sofa", "train", "tvmonitor"]
    
def get_color(c, x, max_val):
    colors = [[1,0,1], [0,0,1], [0,1,1], [0,1,0], [1,1,0], [1,0,0]]
    ratio = float(x)/max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
    return int(r*255)

def draw_bbox(ax, boxes, img_dim, class_id=None):
    """
    boxes: (x_center, y_center, width, height)
    img_dim: (W, H)
    output: plot boxes on img
    """
    for i, box in enumerate(boxes):
        x1 = (box[0] - box[2]/2.0)*img_dim[0]
        y1 = (box[1] - box[3]/2.0)*img_dim[1]
        # x2 = (box[0] + box[2]/2.0)*img_dim[0]
        # y2 = (box[1] + box[3]/2.0)*img_dim[1]         
        width = box[2]*img_dim[0]
        height = box[3]*img_dim[1]        
        
        if class_id[i] is not None:
            classes = len(class_id)
            offset = class_id[i] * 123457 % classes
            red   = get_color(2, offset, classes) / 255
            green = get_color(1, offset, classes) / 255
            blue  = get_color(0, offset, classes) / 255
            color = (red, green, blue)

            ax.text(x1, y1, CLASS_LIST[int(class_id[i])], fontsize=16, color=color)        
        else:
            color = 'r'
        import matplotlib.patches as patches
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
def filter_zero_from_labels(labels):
    labels_new = np.asarray([i for i in labels if np.sum(i) != 0])
    return labels_new

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names