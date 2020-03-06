import torch
import numpy as np
import matplotlib.pyplot as plt

class Average_Meter(object):
    """ computes and stores the average and current value """
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

def mean_L2_error(true, pred):
    """ calculate mean L2 error """
    return np.mean(np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8))

def calc_auc(error_vals, thresholds):
    """ calculate area under curve """
    integral = np.trapz(error_vals, thresholds)
    norm = np.trapz(np.ones_like(error_vals), thresholds)
    return integral/norm

def percentage_frames_within_error_curve(true, pred, max_x=85,
                                         steps=5, plot=True):
    """ caclulate percentage of keypoints within a given error threshold and plot """
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

def get_pck(error_21_pts, max_x=85, steps=5):
    """ caclulate percentage of keypoints within a given error threshold """
    error_threshold = np.arange(0, max_x, steps)

    pck_curve_all = []
    for p_id in range(21):
        pck_curve = []
        for t in error_threshold:
            data_mean = np.mean((error_21_pts[:, p_id] <= t).astype('float32'))
            pck_curve.append(data_mean)
        pck_curve_all.append(pck_curve)
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)
    return pck_curve_all
