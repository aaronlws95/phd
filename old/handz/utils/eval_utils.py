import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# ERROR
# ========================================================

def mean_L2_error(true, pred):
    """
    true: GT points (-1,21,3)
    pred: pred points (-1, 21, 3)
    output: mean L2 distance error
    """
    return np.mean(np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8)) 

def calc_auc(y, x):
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)
    return integral / norm

def percentage_frames_within_error_curve(true, pred, max_x=85, steps=5, plot=True):
    """
    true: GT points (-1,21,3)
    pred: pred points (-1, 21, 3)
    output: PCK curve values
    """
    data_21_points = np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8 )
    error_threshold = np.arange(0, max_x, steps)

    pck_curve_all = []
    for part_id in range(21):
        pck_curve = []
        for t in error_threshold:
            pck_curve.append(np.mean((data_21_points[:, part_id] <= t).astype('float32')))
        pck_curve_all.append(pck_curve)
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(error_threshold,pck_curve_all)
        ax.set_xticks(np.arange(0, max_x, steps))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        plt.grid()
        ax.set_ylabel('Frames within threshold %')
        ax.set_xlabel('Error Threshold mm')
        plt.show()

    return pck_curve_all

def get_pck(data_21_points, max_x=85, steps=5):
    """
    data_points: list of 21 euclidean error values corresponding to 1 hand
    output: PCK curve values
    """

    error_threshold = np.arange(0, max_x, steps)

    pck_curve_all = []
    for part_id in range(21):
        pck_curve = []
        for t in error_threshold:
            pck_curve.append(np.mean((data_21_points[:, part_id] <= t).astype('float32')))
        pck_curve_all.append(pck_curve)
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)

    return pck_curve_all