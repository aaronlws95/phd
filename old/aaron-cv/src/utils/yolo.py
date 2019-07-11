import os
import torch
import time
import struct
import imghdr
import numpy    as np

from src.utils import VOC

# ========================================================
# UTILITIES
# ========================================================

def get_class_labels(set):
    if set == "VOC":
        return VOC.CLASS_LIST
    else:
        raise ValueError(f"no labels for {set}")

def load_darknet_weights(model, weights):
    """ Load darknet yolov2 style weights """
    TFN = torch.from_numpy
    # Parses and loads the weights stored in 'weights'
    weights_file = weights.split(os.sep)[-1]

    # Open the weights file
    with open(weights, 'rb') as f:
        # First 4 are headers
        header  = np.fromfile(f, dtype=np.int32, count=4)
        # The rest are weights
        weights = np.fromfile(f, dtype=np.float32)

    ptr = 0
    for i, (module_def, module) in enumerate(zip(model.module_defs,
                                                 model.module_list)):
        if ptr >= weights.size:
            break
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                # Number of biases
                num_b   = bn_layer.bias.numel()
                # Bias
                bn_b    = TFN(weights[ptr:ptr + num_b])
                bn_b    = bn_b.view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w    = TFN(weights[ptr:ptr + num_b])
                bn_w    = bn_w.view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running mean
                bn_rm   = TFN(weights[ptr:ptr + num_b])
                bn_rm   = bn_rm.view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running var
                bn_rv   = TFN(weights[ptr:ptr + num_b])
                bn_rv   = bn_rv.view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv bias
                num_b   = conv_layer.bias.numel()
                conv_b  = TFN(weights[ptr:ptr + num_b])
                conv_b  = conv_b.view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b

            # Load conv weights
            num_w   = conv_layer.weight.numel()
            conv_w  = torch.from_numpy(weights[ptr:ptr + num_w])
            conv_w  = conv_w.view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

def multi_bbox_iou(boxes1, boxes2):
    """
    Calculate bounding box iou for multiple boxes
    Args:
        boxes: [x_cen, y_cen, w, h] (4, -1)
    Out:
        ious: (-1)
    """
    mx          = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
    Mx          = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
    my          = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
    My          = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
    w1          = boxes1[2]
    h1          = boxes1[3]
    w2          = boxes2[2]
    h2          = boxes2[3]
    uw          = Mx - mx
    uh          = My - my
    cw          = w1 + w2 - uw
    ch          = h1 + h2 - uh
    mask        = ((cw <= 0) + (ch <= 0) > 0)
    area1       = w1 * h1
    area2       = w2 * h2
    carea       = cw * ch
    carea[mask] = 0
    uarea       = area1 + area2 - carea
    return carea/uarea

def bbox_iou(box1, box2):
    """
    Calculate bounding box iou
    Args:
        box: [x_cen, y_cen, w, h] (4)
    Out:
        iou: (1)
    """
    mx      = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
    Mx      = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
    my      = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
    My      = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
    w1      = box1[2]
    h1      = box1[3]
    w2      = box2[2]
    h2      = box2[3]
    uw      = Mx - mx
    uh      = My - my
    cw      = w1 + w2 - uw
    ch      = h1 + h2 - uh
    carea   = 0

    if cw <= 0 or ch <= 0:
        return 0.0

    area1   = w1 * h1
    area2   = w2 * h2
    carea   = cw * ch
    uarea   = area1 + area2 - carea
    return carea/uarea

def get_region_boxes(pred, conf_thresh, nc, anchors, na,
                     only_objectness=True, is_predict=False, is_cuda=False,
                     is_time=False):
    """
    Get boxes with sufficient confidence level from all predicted boxes
    Args:
        pred            : Network output (b, na*(5+nc),
                                          img_w/32, img_h/32)
        conf_thresh     : Confidence threshold
        nc              : Total number of classes
        anchors         : Anchors given as [x1, y1, x2, y2, ...]
        na              : Total number of anchors
        only_objectness : Calculation without accounting for classes
        is_predict      : Is prediction being carried out (adds class to box)
        is_cuda         : Is GPU being used
        is_time         : Displays speed of function
        is_debug        : For debugging
    Out:
        all_boxes       : List of list of boxes [x_cen, y_cen, w, h, det_conf,
                          cls_conf, cls_id, ...]. Add other classes if doing
                          prediction
    """
    if nc == 0 and not only_objectness:
        raise ValueError("nc must be > 0 if only_objectness is False")

    FT                          = torch.FloatTensor
    bs                          = pred.shape[0]
    W                           = pred.shape[2]
    H                           = pred.shape[3]
    anchor_step                 = len(anchors)//na
    t0                          = time.time() # start
    pred                        = pred.view((bs, na, nc + 5, H, W))
    pred_boxes                  = pred[:, :, :4, :, :]
    pred_conf                   = torch.sigmoid(pred[:, :, 4, :, :])
    pred_cls                    = pred[:, :, 5:, :, :]
    pred_boxes[:, :, :2, :, :]  = torch.sigmoid(pred_boxes[:, :, :2, :, :])

    yv, xv                      = torch.meshgrid([torch.arange(H),
                                                  torch.arange(W)])
    grid_x                      = xv.repeat((1, na, 1, 1)).type(FT)
    grid_y                      = yv.repeat((1, na, 1, 1)).type(FT)
    anc                         = torch.Tensor(anchors).view(na, anchor_step)
    anc_w                       = anc[:, 0].view(1, na, 1, 1).type(FT)
    anc_h                       = anc[:, 1].view(1, na, 1, 1).type(FT)

    if is_cuda:
        grid_x      = grid_x.cuda()
        grid_y      = grid_y.cuda()
        anchor_w    = anchor_w.cuda()
        anchor_h    = anchor_h.cuda()

    pred_boxes[:, :, 0, :, :]   = pred_boxes[:, :, 0, :, :] + grid_x
    pred_boxes[:, :, 1, :, :]   = pred_boxes[:, :, 1, :, :] + grid_y
    pred_boxes[:, :, 2, :, :]   = \
        torch.exp(pred_boxes[:, :, 2, :, :])*anc_w
    pred_boxes[:, :, 3, :, :]   = \
        torch.exp(pred_boxes[:, :, 3, :, :])*anc_h

    if nc > 0:
        cls_confs                   = torch.nn.Softmax(dim=2)(pred_cls)
        cls_max_confs, cls_max_ids  = torch.max(cls_confs, 2)
    
    t1 = time.time() # matrix computation

    all_boxes = []
    for b in range(bs):
        # Each batch has a list of boxes for each box with conf > conf_thresh
        boxes = []
        for cy in range(H):
            for cx in range(W):
                for i in range(na):
                    det_conf = pred_conf[b, i, cy, cx]
                    conf = det_conf if only_objectness else \
                        det_conf*cls_max_confs[b, i, cy, cx]
                    # Keep box if more than confidence threshold
                    if conf > conf_thresh:
                        if nc > 0:
                            box = [pred_boxes[b, i, 0, cy, cx]/W,
                                   pred_boxes[b, i, 1, cy, cx]/H,
                                   pred_boxes[b, i, 2, cy, cx]/W,
                                   pred_boxes[b, i, 3, cy, cx]/H,
                                   det_conf,
                                   cls_max_confs[b, i, cy, cx],
                                   cls_max_ids[b, i, cy, cx]]
                        else:
                            box = [pred_boxes[b, i, 0, cy, cx]/W,
                                   pred_boxes[b, i, 1, cy, cx]/H,
                                   pred_boxes[b, i, 2, cy, cx]/W,
                                   pred_boxes[b, i, 3, cy, cx]/H,
                                   det_conf]
                        if not only_objectness and is_predict:
                            for c in range(nc):
                                tmp_conf = cls_confs[b, i, c, cy, cx]
                                if c != cls_max_ids[b, i, cy, cx] and \
                                        det_conf*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t2 = time.time() # end

    if is_time:
        print('-----get_region_boxes-----')
        print('matrix computation : %f' % (t1-t0))
        print('      boxes filter : %f' % (t2-t1))
        print('             total : %f' % (t2-t0))
        print('---------------------------------')

    return all_boxes

def nms_torch(boxes, nms_thresh):
    """
    Non maximum suppresion for torch tensors
    Args:
        boxes:      [x_cen, y_cen, w, h, conf, ...cls] (5+)
        nms_thresh: Non max threshold
    Out:
        out_boxes:  List of boxes same shape as input boxes
    """
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]

    # Sort detection confidence in descending order
    _, sortIds = torch.sort(det_confs)
    out_boxes = []

    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                # filter boxes that overlap with cur box
                if bbox_iou(box_i, box_j) > nms_thresh:
                    box_j[4] = 0
    return out_boxes

def get_image_size(fname):
    """ Determine the image type of fhandle and return its size """
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                # Read 0xff next
                fhandle.seek(0)
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                # Skip `precision' byte.
                fhandle.seek(1, 1)
                height, width = struct.unpack('>HH', fhandle.read(4))
            # IGNORE:W0703
            except Exception:
                return
        else:
            return
        return width, height

def get_num_gt(target):
    """
    Get number of GT from target
    Args:
        target: (max_boxes=50, 5) [cls, x, y, w, h]
    Out:
        i: Number of ground truths
    """
    for i in range(50):
        if target[i][1] == 0:
            return i

    raise ValueError("Target should have 0 value")

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