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

def get_pred_gt(epoch, data_split, exp):
    data_file_h5py = os.path.join(DATA_DIR, '%s_fpha_RGB.h5' %data_split)
    pred_file = os.path.join(DATA_DIR, exp, 'predict_%s_%s.txt' %(epoch, data_split))
    img0, img1, img2, uvd_norm_gt, uvd_gt, xyz_gt, hand_center_uvd, file_name = pd.read_data_h5py(data_file_h5py)
    pred_xyz, pred_uvd, pred_normuvd = pd.get_pred_xyzuvd_from_normuvd(pred_file, hand_center_uvd)
    xyz_gt = np.reshape(xyz_gt, (-1, 63))
    uvd_gt = np.reshape(uvd_gt, (-1, 63))
    uvd_norm_gt = np.reshape(uvd_norm_gt, (-1, 63))
    pred_xyz = np.reshape(pred_xyz, (-1, 63))
    pred_uvd = np.reshape(pred_uvd, (-1, 63))
    pred_normuvd = np.reshape(pred_normuvd, (-1, 63))
    return xyz_gt, pred_xyz, uvd_gt, pred_uvd, uvd_norm_gt, pred_normuvd

def epoch_xyz_loss_curve(epochs, exp):
    data_file_h5py_train = os.path.join(DATA_DIR, 'train_fpha_RGB.h5')
    data_file_h5py_test = os.path.join(DATA_DIR, 'test_fpha_RGB.h5')

    _, _, _, _, _, xyz_gt_train, hand_center_uvd_train, _ = read_data_h5py(data_file_h5py_train)
    xyz_gt_train = np.reshape(xyz_gt_train, (-1, 63))

    _, _, _, _, _, xyz_gt_test, hand_center_uvd_test, _ = read_data_h5py(data_file_h5py_test)
    xyz_gt_test = np.reshape(xyz_gt_test, (-1, 63))

    xyz_loss_train = []
    xyz_loss_test = []
    for epoch in epochs:
        pred_file_train = os.path.join(DATA_DIR, exp, 'predict_%s_train.txt' %epoch)
        pred_xyz_train, _, _ = pd.get_pred_xyzuvd_from_normuvd(pred_file_train, hand_center_uvd_train)
        pred_xyz_train = np.reshape(pred_xyz_train, (-1, 63))
        xyz_loss_train.append(mean_pose_error(pred_xyz_train, xyz_gt_train[:pred_xyz_train.shape[0]]))

        pred_file_test = os.path.join(DATA_DIR, exp, 'predict_%s_test.txt' %epoch)
        pred_xyz_test, _, _ = pd.get_pred_xyzuvd_from_normuvd(pred_file_test, hand_center_uvd_test)
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

def mean_squared_error(true, pred):
    return np.mean(np.square(pred - true), axis=-1)

def mean_pose_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.mean(np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1)))

def pose_error(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.mean(np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1)), axis=-1)

def pose_error_21(true, pred):
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    return np.mean(np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1)), axis=0)

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
    true_3dim = np.reshape(true, (-1,21,3))
    pred_3dim = np.reshape(pred, (-1,21,3))
    data_21 = np.sqrt(np.sum(np.square(true_3dim-pred_3dim), axis=-1))
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
