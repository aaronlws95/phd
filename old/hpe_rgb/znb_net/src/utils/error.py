import numpy as np
import os
import matplotlib.pyplot as plt
import math

from .directory import DATA_DIR
import utils.prepare_data as pd

def calc_auc(y, x):
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)
    return integral / norm

def scoremap_error(true, pred):
    return np.mean(np.sqrt(np.sum(np.square(true-pred), axis=0) + 1e-8 ))

def mean_pose_error(true, pred):
    return np.mean(np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8 ))

def percentage_frames_within_error_curve(true, pred, max_x=100):
    frame_error = pose_error(true, pred)
    # max_error = int(math.ceil(np.max(frame_error)/100))*100
    error_threshold = np.arange(0, max_x, max_x//10)
    perc_frames_within_error = [((np.sum(frame_error < e))/len(frame_error))*100 for e in error_threshold]
    fig, ax = plt.subplots()
    ax.plot(error_threshold,perc_frames_within_error)
    ax.set_xticks(np.arange(0, max_x, max_x//10))
    ax.set_yticks(np.arange(0, 110, 10))
    plt.grid()
    ax.set_ylabel('Frames within threshold %')
    ax.set_xlabel('Error Threshold mm')
    plt.show()

def percentage_frames_within_error_curve_zimmmerman(true, pred, max_x=85, steps=5):
    # true_3dim = np.reshape(true, (-1,21,3))
    # pred_3dim = np.reshape(pred, (-1,21,3))
    data_21 = np.sqrt(np.sum(np.square(true-pred), axis=-1) + 1e-8 )
    error_threshold = np.arange(0, max_x, steps)

    pck_curve_all = []
    for part_id in range(21):
        pck_curve = []
        for t in error_threshold:
            pck_curve.append(np.mean((data_21[:, part_id] <= t).astype('float')))
        pck_curve_all.append(pck_curve)
    pck_curve_all = np.mean(np.array(pck_curve_all), 0)

    fig, ax = plt.subplots()
    ax.plot(error_threshold,pck_curve_all)
    ax.set_xticks(np.arange(0, max_x, steps))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.grid()
    ax.set_ylabel('Frames within threshold %')
    ax.set_xlabel('Error Threshold mm')
    plt.show()

    return pck_curve_all
