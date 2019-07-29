import torch

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

    area1   = w1*h1
    area2   = w2*h2
    carea   = cw*ch
    uarea   = area1 + area2 - carea
    return carea/uarea

def get_region_boxes(pred, conf_thresh, nc, anchors, na,
                     only_objectness=True, is_predict=False, is_cuda=False):
    """
    Get boxes with sufficient confidence level from all predicted boxes
    Args:
        pred            : Network output (b, na*(5+nc),
                                          img_h/32, img_w/32)
        conf_thresh     : Confidence threshold
        nc              : Total number of classes
        anchors         : Anchors given as [x1, y1, x2, y2, ...]
        na              : Total number of anchors
        only_objectness : Calculation without accounting for classes
        is_predict      : Is prediction being carried out (adds class to box)
        is_cuda         : Is GPU being used
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
    H                           = pred.shape[2]
    W                           = pred.shape[3]
    anchor_step                 = len(anchors)//na
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
        det_confs[i] = 1 - boxes[i][4]

    # Sort detection confidence in descending order
    _, sortIds = torch.sort(det_confs)
    out_boxes = []

    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                # filter boxes that overlap with cur box
                if bbox_iou(box_i, box_j) > nms_thresh:
                    box_j[4] = 0
    return out_boxes