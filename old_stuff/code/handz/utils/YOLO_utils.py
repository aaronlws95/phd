import os
import sys
import torch
import numpy as np 
from PIL import Image
import time

import utils.VOC_utils as VOC
import utils.COCO_utils as COCO
from utils.image_utils import *
from utils.dir import dir_dict

DIR = dir_dict["YOLO_DIR"]
FPHA_DIR = dir_dict["YOLO_FPHA_DIR"]

# ========================================================
# MISC
# ========================================================
    
def get_class_labels(set):
    if set == "VOC":
        return VOC.CLASS_LIST
    elif set == "COCO":
        return COCO.CLASS_LIST
    else:
        raise ValueError(f"no labels for {set}")

# ========================================================
# ULTRALYTICS
# ========================================================

def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou

def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords

def non_max_suppression(prediction, conf_thres, nms_thres):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """
    # hyperlytics
    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & (torch.isnan(pred).any(1) == 0)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # No NMS required if only 1 prediction
            if len(dc) == 1:
                det_max.append(dc)
                continue

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou_hyperlytics(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou_hyperlytics(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou_hyperlytics(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output

def non_max_suppression_1class(prediction, conf_thres, nms_thres):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf)
    """
    # hyperlytics
    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & (torch.isnan(pred).any(1) == 0)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)

        dc = pred
        dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

        # No NMS required if only 1 prediction
        if len(dc) == 1:
            det_max.append(dc)
            continue

        # Non-maximum suppression
        if nms_style == 'OR':  # default
            # METHOD1
            # ind = list(range(len(dc)))
            # while len(ind):
            # j = ind[0]
            # det_max.append(dc[j:j + 1])  # save highest conf detection
            # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
            # [ind.pop(i) for i in reversed(reject)]

            # METHOD2
            while dc.shape[0]:
                det_max.append(dc[:1])  # save highest conf detection
                if len(dc) == 1:  # Stop if we're at the last detection
                    break
                iou = bbox_iou_hyperlytics(dc[0], dc[1:])  # iou with other boxes
                dc = dc[1:][iou < nms_thres]  # remove ious > threshold

        elif nms_style == 'AND':  # requires overlap, single boxes erased
            while len(dc) > 1:
                iou = bbox_iou_hyperlytics(dc[0], dc[1:])  # iou with other boxes
                if iou.max() > 0.5:
                    det_max.append(dc[:1])
                dc = dc[1:][iou < nms_thres]  # remove ious > threshold

        elif nms_style == 'MERGE':  # weighted mixture box
            while len(dc):
                if len(dc) == 1:
                    det_max.append(dc)
                    break
                i = bbox_iou_hyperlytics(dc[0], dc) > nms_thres  # iou with other boxes
                weights = dc[i, 4:5]
                dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                det_max.append(dc[:1])
                dc = dc[i == 0]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # Plot
            # plt.plot(recall_curve, precision_curve)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def bbox_iou_hyperlytics(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou

def xyxy2xywh(x):
    """"
    x: (-1, 4)
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    """
    
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def xywh2xyxy(x):
    """
    x: (-1, 4)
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# ========================================================
# MARVIS
# ========================================================

def get_num_gt(target):
    """
    get number of gt from target
    target: (max_boxes=50, 5)
    """
    for i in range(50):
        if target[i][1] == 0:
            return i
    
    raise ValueError("target should have 0 value")

def nms_torch(boxes, nms_thresh):
    """
    non maximum suppresion for torch tensors
    boxes: [x, y, w, h, conf, ...]
    """    
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    # sortIds in descending order
    _,sortIds = torch.sort(det_confs)
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

def multi_bbox_iou(boxes1, boxes2):
    """
    input x,y,w,h (4, -1)
    """
    mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
    Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
    my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
    My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
    w1 = boxes1[2]
    h1 = boxes1[3]
    w2 = boxes2[2]
    h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea    

def bbox_iou(box1, box2):
    """
    input: x,y,w,h (4)
    """
    mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
    Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
    my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
    My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
    w1 = box1[2]
    h1 = box1[3]
    w2 = box2[2]
    h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea    

def get_image_size(fname):
    """
    Determine the image type of fhandle and return its size.
    from draco
    """
    import struct
    import imghdr 
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
                fhandle.seek(0) # Read 0xff next
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
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height    

def get_region_boxes(pred, conf_thresh, num_classes, anchors, num_anchors, only_objectness=True, is_predict=False, is_cuda=False, is_time=False):

    if num_classes == 0 and not only_objectness:
        raise ValueError("num_classes must be > 0 if only_objectness is False")

    batch_size = pred.shape[0] 
    W = pred.shape[2]
    H = pred.shape[3]        
    anchor_step = len(anchors) // num_anchors
    
    t0 = time.time() # start time
    pred = pred.view(batch_size*num_anchors, 5+num_classes, H*W).transpose(0,1).contiguous().view(5+num_classes, batch_size*num_anchors*H*W)

    grid_x = torch.linspace(0, W-1, W).repeat(H,1).repeat(batch_size*num_anchors, 1, 1).view((batch_size*num_anchors*W*H))
    grid_y = torch.linspace(0, H-1, H).repeat(W,1).t().repeat(batch_size*num_anchors, 1, 1).view((batch_size*num_anchors*W*H))
    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, H*W).view((batch_size*num_anchors*W*H))
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, H*W).view((batch_size*num_anchors*W*H))
    if is_cuda:
        grid_x  = grid_x.cuda()
        grid_y  = grid_y.cuda() 
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()
        
    xs = torch.sigmoid(pred[0]) + grid_x
    ys = torch.sigmoid(pred[1]) + grid_y
    ws = torch.exp(pred[2]) * anchor_w
    hs = torch.exp(pred[3]) * anchor_h
    confs = torch.sigmoid(pred[4])
    
    if num_classes > 0:
        cls_confs = torch.nn.Softmax(dim=0)(pred[5:])
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 0)
    
    t1 = time.time() # matrix computation end
    all_boxes = []
    for b in range(batch_size):
        boxes = []
        for cy in range(H):
            for cx in range(W):
                for i in range(num_anchors):
                    ind = b*H*W*num_anchors + i*H*W+ cy*W + cx
                    detection_conf = confs[ind]
                    if only_objectness:
                        conf = detection_conf
                    else:
                        conf = detection_conf * cls_max_confs[ind]
                    # keep box if more than confidence threshold
                    if conf > conf_thresh:
                        if num_classes > 0:
                            box = [xs[ind]/W, ys[ind]/H, ws[ind]/W, hs[ind]/H, detection_conf, cls_max_confs[ind], cls_max_ids[ind]]
                        else:
                            box = [xs[ind]/W, ys[ind]/H, ws[ind]/W, hs[ind]/H, detection_conf]
                        if not only_objectness and is_predict:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[c, ind]
                                if c != cls_max_ids[ind] and detection_conf*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        t6 = time.time()                                 
                        boxes.append(box)   
        all_boxes.append(boxes)
    t2 = time.time() # end time
    
    if is_time:
        print('-----get_region_boxes-----')
        print('matrix computation : %f' % (t1-t0))
        print('      boxes filter : %f' % (t2-t1))
        print('             total : %f' % (t2-t0))
        print('---------------------------------')  
    
    return all_boxes

def detect(model, img, conf_thresh, nms_thresh, is_cuda=True):
    """
    Carry out detection
    output: x_center, y_center, w, h
    """
    if type(img) == np.ndarray:
        img = (img / 255.0)
        img = imgshape2torch(img)
        img = np.expand_dims(img, 0)
        img = torch.FloatTensor(img)
    else:
        raise ValueError("img must be numpy array")
    
    if is_cuda:
        img = img.cuda()
    
    model.net.eval()
    with torch.no_grad():
        pred = model.net(img)

    boxes = get_region_boxes(pred, conf_thresh, model.num_classes, model.anchors, model.num_anchors, is_cuda=is_cuda)[0]
    boxes = nms_torch(boxes, nms_thresh)
    
    boxes_cpu = []
    for box in boxes:
        boxes_cpu.append([float(i) for i in box])
    
    return boxes_cpu

# ========================================================
# EVALUATION
# ========================================================
#https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
#https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_dataset_evaluator.py

import xml.etree.ElementTree as ET
import pickle

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             data_split,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_{}.pkl'.format(data_split))
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def do_python_voc_eval(res_prefix, devkit_path, class_list, year, data_split):
    classes = tuple(['__background__'] + class_list)
    filename = res_prefix + '{:s}.txt'
    annopath = os.path.join(
        devkit_path,
        'VOC' + year,
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC' + year,
        'ImageSets',
        'Main',
        data_split + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    for cls in classes:
        if cls == '__background__':
            continue
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, data_split, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    # print('-- Thanks, The Management')
    # print('--------------------------------------------------------------')
    