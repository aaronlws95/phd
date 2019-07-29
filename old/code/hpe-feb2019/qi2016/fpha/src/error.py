import numpy as np
import os
import matplotlib.pyplot as plt
import math

import constants
import utils
from prepare_data import prepare_data, read_hand_center_uvd, read_prepare_data_h5py

def epoch_xyz_loss_curve(epochs, exp):
    data_file_h5py_train = os.path.join(constants.DATA_DIR, 'train_fpha.h5')
    data_file_h5py_test = os.path.join(constants.DATA_DIR, 'test_fpha.h5')

    _, _, _, _, _, xyz_gt_train, hand_center_uvd_train, _ = read_prepare_data_h5py(data_file_h5py_train)
    xyz_gt_train = np.reshape(xyz_gt_train, (-1, 63))

    _, _, _, _, _, xyz_gt_test, hand_center_uvd_test, _ = read_prepare_data_h5py(data_file_h5py_test)
    xyz_gt_test = np.reshape(xyz_gt_test, (-1, 63))

    xyz_loss_train = []
    xyz_loss_test = []
    for epoch in epochs:
        pred_file_train = os.path.join(constants.DATA_DIR, exp, 'predict_%s_train.txt' %epoch)
        pred_xyz_train, _, _ = get_pred_xyzuvd_from_normuvd(pred_file_train, hand_center_uvd_train)
        pred_xyz_train = np.reshape(pred_xyz_train, (-1, 63))
        xyz_loss_train.append(mean_pose_error(pred_xyz_train, xyz_gt_train[:pred_xyz_train.shape[0]]))

        pred_file_test = os.path.join(constants.DATA_DIR, exp, 'predict_%s_test.txt' %epoch)
        pred_xyz_test, _, _ = get_pred_xyzuvd_from_normuvd(pred_file_test, hand_center_uvd_test)
        pred_xyz_test = np.reshape(pred_xyz_test, (-1, 63))
        xyz_loss_test.append(mean_pose_error(pred_xyz_test, xyz_gt_test[:pred_xyz_test.shape[0]]))

    fig, ax = plt.subplots()
    ax.plot(epochs,xyz_loss_train, color='red')
    ax.plot(epochs,xyz_loss_test, color='blue')
    # ax.set_xticks(np.arange(0, max_x, max_x//10))
    # ax.set_yticks(np.arange(0, 110, 10))
    plt.grid()
    ax.set_ylabel('Pose error/mm')
    ax.set_xlabel('Epoch')
    plt.show()

def read_predict(pred_file):
    pred_normuvd = []
    with open(pred_file, "r") as f:
        pred_line = f.readlines()
        for lines in pred_line:
            pred_normuvd.append([float(i) for i in lines.strip().split()])
    return pred_normuvd

def get_pred_xyzuvd_from_normuvd(pred_file, hand_center_uvd):
    pred_normuvd = np.reshape(read_predict(pred_file), (-1, 21, 3))
    pred_xyz, pred_uvd = utils.normuvd2xyzuvd_batch_depth(pred_normuvd, hand_center_uvd[:pred_normuvd.shape[0]])
    return np.reshape(pred_xyz, (-1, 63)), np.reshape(pred_uvd, (-1, 63)), np.reshape(pred_normuvd, (-1, 63))

def mean_squared_error(true, pred):
    return np.mean(np.square(pred - true), axis=-1)

def mean_overall_squared_error(true, pred):
    return np.mean(np.mean(np.square(pred - true), axis=-1))

def l2_dist(true, pred):
    err = np.sqrt(np.sum(np.square(true-pred), axis=-1))
    return err

def dim_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=1))

def jnt_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1))

def overall_error(true, pred):
    return np.sqrt(np.sum(np.square(true-pred), axis=1))

def mean_dim_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.mean(np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=1)), axis=0)

def mean_jnt_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.mean(np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1)), axis=0)

def mean_jnt_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.mean(np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1)), axis=0)

def mean_2D_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    true_2dim = np.reshape(true_3dim[:, :, :2], (-1, 42))
    pred_2dim = np.reshape(pred_3dim[:, :, :2], (-1, 42))
    return  np.mean(np.sqrt(np.sum(np.square(true_2dim-pred_2dim), axis=1)), axis=0)

def mean_overall_error(true, pred):
    return  np.mean(np.sqrt(np.sum(np.square(true-pred), axis=1)), axis=0)

def mean_pose_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.mean(np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1)))

def pose_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.mean(np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1)), axis=-1)

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
