from PIL import Image
from matplotlib import pyplot as plt 
import os
import math 
import numpy as np

from utils.dir import dir_dict

DIR = dir_dict["COCO_DIR"]
    
CLASS_LIST = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
              "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
              "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
              "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", 
              "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", 
              "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
              "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
              "scissors", "teddy bear", "hair drier", "toothbrush"]

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

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