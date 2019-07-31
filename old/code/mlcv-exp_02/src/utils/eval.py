import torch
import numpy as np
import matplotlib.pyplot as plt

class Average_Meter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val    = 0
        self.avg    = 0
        self.sum    = 0
        self.count  = 0

    def update(self, val, n):
        self.val    = val
        self.sum    += val*n
        self.count  += n
        self.avg    = self.sum/self.count

def topk_accuracy(output, target, topk):
    """ Computes the precision@k for the specified values of k """
    maxk        = max(topk)
    batch_size  = target.size(0)
    _, pred     = output.topk(maxk, 1, True, True)
    pred        = pred.t()
    correct     = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def mean_class_accuracy(pred_dist, all_gt):
    num_class = len(np.unique(all_gt))
    correct_cls = np.zeros(num_class)
    total_cls = np.zeros(num_class)
    for pred, gt in zip(pred_dist, all_gt):
        pred_cls = np.argmax(pred)
        if pred_cls == gt:
            correct_cls[pred_cls] += 1
        total_cls[gt] += 1
    return np.mean(correct_cls/(total_cls + 1e-8))

def mean_class_accuracy_topk(pred_dist, all_gt, k):
    num_class = len(np.unique(all_gt))
    correct_cls = np.zeros(num_class)
    total_cls = np.zeros(num_class)
    for pred, gt in zip(pred_dist, all_gt):
        pred_cls = np.argsort(pred)[::-1][:k]
        if gt in pred_cls:
            correct_cls[gt] += 1
        total_cls[gt] += 1
    return np.mean(correct_cls/(total_cls + 1e-8))

def mean_class_accuracy_2(pred_dist_1, pred_dist_2,
                          all_gt_1, all_gt_2, num_class_1, num_class_2):
    correct_cls_1 = np.zeros(num_class_1)
    total_cls_1 = np.zeros(num_class_1)
    correct_cls_2 = np.zeros(num_class_2)
    total_cls_2 = np.zeros(num_class_2)

    class_both = list(set(zip(all_gt_1, all_gt_2)))
    correct_cls_both = {i: 0 for i in class_both}
    total_cls_both = {i: 0 for i in class_both}
    for pred_1, pred_2, gt_1, gt_2 in zip(pred_dist_1, pred_dist_2,
                                          all_gt_1, all_gt_2):
        pred_1_cls = np.argmax(pred_1)
        pred_2_cls = np.argmax(pred_2)

        if pred_1_cls == gt_1:
            correct_cls_1[pred_1_cls] += 1
        if pred_2_cls == gt_2:
            correct_cls_2[pred_2_cls] += 1
        if pred_1_cls == gt_1 and pred_2_cls == gt_2:
            correct_cls_both[(gt_1, gt_2)] += 1

        total_cls_1[gt_1] += 1
        total_cls_2[gt_2] += 1
        if (gt_1, gt_2) in total_cls_both:
            total_cls_both[(gt_1, gt_2)] += 1

    acc_1 = np.mean(correct_cls_1/(total_cls_1 + 1e-8))
    acc_2 = np.mean(correct_cls_2/(total_cls_2 + 1e-8))
    acc_both = np.mean([correct_cls_both[k]/(total_cls_both[k] + 1e-8) for k in total_cls_both.keys()])

    return acc_1, acc_2, acc_both

def seq_mean_class_accuracy_2(seq_info, pred_dist_1, pred_dist_2,
                          all_gt_1, all_gt_2, consensus, num_class_1, num_class_2):
    correct_cls_1 = np.zeros(num_class_1)
    total_cls_1 = np.zeros(num_class_1)
    correct_cls_2 = np.zeros(num_class_2)
    total_cls_2 = np.zeros(num_class_2)

    class_both = list(set(zip(all_gt_1, all_gt_2)))
    correct_cls_both = {i: 0 for i in class_both}
    total_cls_both = {i: 0 for i in class_both}

    track = 0
    for pl, act_gt, obj_gt in seq_info:
        pred_1 = pred_dist_1[track:track+pl]
        pred_2 = pred_dist_2[track:track+pl]

        if consensus == 'mean_dist':
            pred_1 = np.mean(pred_1, 0)
            pred_2 = np.mean(pred_2, 0)
            pred_1_cls = np.argmax(pred_1)
            pred_2_cls = np.argmax(pred_2)
        elif consensus == 'pred_freq':
            pred_1 = np.bincount(np.argmax(pred_1, 1))
            pred_2 = np.bincount(np.argmax(pred_2, 1))
            pred_1_cls = np.argmax(pred_1)
            pred_2_cls = np.argmax(pred_2)

        if pred_1_cls == act_gt:
            correct_cls_1[pred_1_cls] += 1
        if pred_2_cls == obj_gt:
            correct_cls_2[pred_2_cls] += 1
        if pred_1_cls == act_gt and pred_2_cls == obj_gt:
            correct_cls_both[(act_gt, obj_gt)] += 1

        total_cls_1[act_gt] += 1
        total_cls_2[obj_gt] += 1
        if (act_gt, obj_gt) in total_cls_both:
            total_cls_both[(act_gt, obj_gt)] += 1

        track += pl
    acc_1 = np.mean(correct_cls_1/(total_cls_1 + 1e-8))
    acc_2 = np.mean(correct_cls_2/(total_cls_2 + 1e-8))
    acc_both = np.mean([correct_cls_both[k]/total_cls_both[k] for k in total_cls_both.keys()])

    return acc_1, acc_2, acc_both

def seq_seg_mean_class_accuracy_2(seq_info, pred_dist_1, pred_dist_2,
                          all_gt_1, all_gt_2, consensus, num_segments, num_class_1, num_class_2):
    correct_cls_1 = np.zeros(num_class_1)
    total_cls_1 = np.zeros(num_class_1)
    correct_cls_2 = np.zeros(num_class_2)
    total_cls_2 = np.zeros(num_class_2)

    class_both = list(set(zip(all_gt_1, all_gt_2)))
    correct_cls_both = {i: 0 for i in class_both}
    total_cls_both = {i: 0 for i in class_both}

    track = 0
    for pl, act_gt, obj_gt in seq_info:
        avg_duration = pl//num_segments
        offset = np.random.randint(avg_duration, size=num_segments)
        offset = np.multiply(list(range(num_segments)), avg_duration) + offset
        frames = [track + o for o in offset]
        pred_1 = pred_dist_1[frames]
        pred_2 = pred_dist_2[frames]

        if consensus == 'mean_dist':
            pred_1 = np.mean(pred_1, 0)
            pred_2 = np.mean(pred_2, 0)
            pred_1_cls = np.argmax(pred_1)
            pred_2_cls = np.argmax(pred_2)
        elif consensus == 'pred_freq':
            pred_1 = np.bincount(np.argmax(pred_1, 1))
            pred_2 = np.bincount(np.argmax(pred_2, 1))
            pred_1_cls = np.argmax(pred_1)
            pred_2_cls = np.argmax(pred_2)

        if pred_1_cls == act_gt:
            correct_cls_1[pred_1_cls] += 1
        if pred_2_cls == obj_gt:
            correct_cls_2[pred_2_cls] += 1
        if pred_1_cls == act_gt and pred_2_cls == obj_gt:
            correct_cls_both[(act_gt, obj_gt)] += 1

        total_cls_1[act_gt] += 1
        total_cls_2[obj_gt] += 1
        if (act_gt, obj_gt) in total_cls_both:
            total_cls_both[(act_gt, obj_gt)] += 1

        track += pl
    acc_1 = np.mean(correct_cls_1/(total_cls_1 + 1e-8))
    acc_2 = np.mean(correct_cls_2/(total_cls_2 + 1e-8))
    acc_both = np.mean([correct_cls_both[k]/total_cls_both[k] for k in total_cls_both.keys()])

    return acc_1, acc_2, acc_both

def mean_class_topk_2(pred_dist_1, pred_dist_2,
                          all_gt_1, all_gt_2, k, num_class_1, num_class_2):
    correct_cls_1 = np.zeros(num_class_1)
    total_cls_1 = np.zeros(num_class_1)
    correct_cls_2 = np.zeros(num_class_2)
    total_cls_2 = np.zeros(num_class_2)

    class_both = list(set(zip(all_gt_1, all_gt_2)))
    correct_cls_both = {i: 0 for i in class_both}
    total_cls_both = {i: 0 for i in class_both}

    for pred_1, pred_2, gt_1, gt_2 in zip(pred_dist_1, pred_dist_2,
                                          all_gt_1, all_gt_2):
        pred_1_cls = np.argsort(pred_1)[::-1][:k]
        pred_2_cls = np.argsort(pred_2)[::-1][:k]

        if gt_1 in pred_1_cls:
            correct_cls_1[gt_1] += 1
        if gt_2 in pred_2_cls:
            correct_cls_2[gt_2] += 1
        if gt_2 in pred_2_cls and gt_1 in pred_1_cls:
            correct_cls_both[(gt_1, gt_2)] += 1

        total_cls_1[gt_1] += 1
        total_cls_2[gt_2] += 1
        if (gt_1, gt_2) in total_cls_both:
            total_cls_both[(gt_1, gt_2)] += 1
    acc_1 = np.mean(correct_cls_1/(total_cls_1 + 1e-8))
    acc_2 = np.mean(correct_cls_2/(total_cls_2 + 1e-8))
    acc_both = np.mean([correct_cls_both[k]/(total_cls_both[k] + 1e-8) for k in total_cls_both.keys()])

    return acc_1, acc_2, acc_both

def mean_accuracy_2(pred_dist_1, pred_dist_2,
                          all_gt_1, all_gt_2):
    correct_1 = 0
    correct_2 = 0
    correct_both = 0
    for pred_1, pred_2, gt_1, gt_2 in zip(pred_dist_1, pred_dist_2,
                                          all_gt_1, all_gt_2):
        pred_1_cls = np.argmax(pred_1)
        pred_2_cls = np.argmax(pred_2)

        if pred_1_cls == gt_1:
            correct_1 += 1
        if pred_2_cls == gt_2:
            correct_2 += 1
        if pred_1_cls == gt_1 and pred_2_cls == gt_2:
            correct_both += 1

    total_len = len(all_gt_1)
    return correct_1/total_len, correct_2/total_len, correct_both/total_len

def mean_topk_2(pred_dist_1, pred_dist_2,
                          all_gt_1, all_gt_2, k):
    correct_1 = 0
    correct_2 = 0
    correct_both = 0
    for pred_1, pred_2, gt_1, gt_2 in zip(pred_dist_1, pred_dist_2,
                                          all_gt_1, all_gt_2):
        pred_1_cls = np.argsort(pred_1)[::-1][:k]
        pred_2_cls = np.argsort(pred_2)[::-1][:k]

        if gt_1 in pred_1_cls:
            correct_1 += 1
        if gt_2 in pred_2_cls:
            correct_2 += 1
        if gt_1 in pred_1_cls and gt_2 in pred_2_cls:
            correct_both += 1

    total_len = len(all_gt_1)
    return correct_1/total_len, correct_2/total_len, correct_both/total_len

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
    return integral/norm

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
