import os
import cv2
import torch
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

def compute_resized_coords(coords, resizeFactor):
    """
    Given the index/coordinates of a cell in some input array (e.g. image),
    provides the new coordinates if that array was resized by making it
    resizeFactor times bigger.
    E.g.: image of size 3x3 is resized to 6x6 (resizeFactor=2), we'd like to
    know the new coordinates of cell [1,2] -> Function would return [2.5,4.5]
    :param coords: Coordinates (indices) of a cell in some input array
    :param resizeFactor: Resize coefficient = shape_dest/shape_source. E.g.:
    resizeFactor=2 means the destination array is twice as big as the
    original one
    :return: Coordinates in an array of size
    shape_dest=resizeFactor*shape_source, expressing the array indices of the
    closest point to 'coords' if an image of size shape_source was resized to
    shape_dest
    """

    # 1) Add 0.5 to coords to get coordinates of center of the pixel (e.g.
    # index [0,0] represents the pixel at location [0.5,0.5])
    # 2) Transform those coordinates to shape_dest, by multiplying by resizeFactor
    # 3) That number represents the location of the pixel center in the new array,
    # so subtract 0.5 to get coordinates of the array index/indices (revert
    # step 1)
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5


def find_peaks(thresh, img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > thresh)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T

def NMS(heatmaps, upsampFactor=1., bool_refine_center=True, bool_gaussian_filt=False, thresh=0.1):
    """
    NonMaximaSuppression: find peaks (local maxima) in a set of grayscale images
    :param heatmaps: set of grayscale images on which to find local maxima (3d np.array,
    with dimensions image_height x image_width x num_heatmaps)
    :param upsampFactor: Size ratio between CPM heatmap output and the input image size.
    Eg: upsampFactor=16 if original image was 480x640 and heatmaps are 30x40xN
    :param bool_refine_center: Flag indicating whether:
     - False: Simply return the low-res peak found upscaled by upsampFactor (subject to grid-snap)
     - True: (Recommended, very accurate) Upsample a small patch around each low-res peak and
     fine-tune the location of the peak at the resolution of the original input image
    :param bool_gaussian_filt: Flag indicating whether to apply a 1d-GaussianFilter (smoothing)
    to each upsampled patch before fine-tuning the location of each peak.
    :return: a NUM_JOINTS x 4 np.array where each row represents a joint type (0=nose, 1=neck...)
    and the columns indicate the {x,y} position, the score (probability) and a unique id (counter)
    """
    # MODIFIED BY CARLOS: Instead of upsampling the heatmaps to heatmap_avg and
    # then performing NMS to find peaks, this step can be sped up by ~25-50x by:
    # (9-10ms [with GaussFilt] or 5-6ms [without GaussFilt] vs 250-280ms on RoG
    # 1. Perform NMS at (low-res) CPM's output resolution
    # 1.1. Find peaks using scipy.ndimage.filters.maximum_filter
    # 2. Once a peak is found, take a patch of 5x5 centered around the peak, upsample it, and
    # fine-tune the position of the actual maximum.
    #  '-> That's equivalent to having found the peak on heatmap_avg, but much faster because we only
    #      upsample and scan the 5x5 patch instead of the full (e.g.) 480x640

    joint_list_per_joint_type = []
    cnt_total_joints = 0

    # For every peak found, win_size specifies how many pixels in each
    # direction from the peak we take to obtain the patch that will be
    # upsampled. Eg: win_size=1 -> patch is 3x3; win_size=2 -> 5x5
    # (for BICUBIC interpolation to be accurate, win_size needs to be >=2!)
    win_size = 2

    for joint in range(heatmaps.shape[-1]):
        map_orig = heatmaps[:, :, joint]
        peak_coords = find_peaks(thresh, map_orig)
        peaks = np.zeros((len(peak_coords), 4))
        for i, peak in enumerate(peak_coords):
            if bool_refine_center:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(
                    np.array(map_orig.T.shape) - 1, peak + win_size)

                # Take a small patch around each peak and only upsample that
                # tiny region
                patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
                map_upsamp = cv2.resize(
                    patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv2.INTER_CUBIC)

                # Gaussian filtering takes an average of 0.8ms/peak (and there might be
                # more than one peak per joint!) -> For now, skip it (it's
                # accurate enough)
                map_upsamp = gaussian_filter(
                    map_upsamp, sigma=3) if bool_gaussian_filt else map_upsamp

                # Obtain the coordinates of the maximum value in the patch
                location_of_max = np.unravel_index(
                    map_upsamp.argmax(), map_upsamp.shape)
                # Remember that peaks indicates [x,y] -> need to reverse it for
                # [y,x]
                location_of_patch_center = compute_resized_coords(
                    peak[::-1] - [y_min, x_min], upsampFactor)
                # Calculate the offset wrt to the patch center where the actual
                # maximum is
                refined_center = (location_of_max - location_of_patch_center)
                peak_score = map_upsamp[location_of_max]
            else:
                refined_center = [0, 0]
                # Flip peak coordinates since they are [x,y] instead of [y,x]
                peak_score = map_orig[tuple(peak[::-1])]
            peaks[i, :] = tuple([int(round(x)) for x in compute_resized_coords(
                peak_coords[i], upsampFactor) + refined_center[::-1]]) + (peak_score, cnt_total_joints)
            cnt_total_joints += 1
        joint_list_per_joint_type.append(peaks)

    return joint_list_per_joint_type

def gen_paf(centerA, centerB, acc_vec_map, count, stride=8, in_size=256):
    """ B --> A """
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    grid_y = in_size//stride
    grid_x = in_size//stride
    
    thre = 1   # limb width
    centerB = centerB/stride
    centerA = centerA/stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm == 0.0):
        return acc_vec_map, count
    limb_vec_unit = limb_vec / norm

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x, y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(acc_vec_map) * 0.0
    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    acc_vec_map = np.multiply(
        acc_vec_map, count[:, :, np.newaxis])
    acc_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    acc_vec_map = np.divide(acc_vec_map, count[:, :, np.newaxis])
    count[mask == True] = 0

    return acc_vec_map, count

def get_paf():
    """ for dataloader **not working currently """
    paf        = np.zeros((self.img_size//self.stride, self.img_size//self.stride, 40))
    paf_vis    = np.ones((self.img_size//self.stride, self.img_size//self.stride, 40))
    jnt_from    = [0, 0, 0, 0, 0, 1, 6, 7, 2,  9, 10,  3, 12, 13,  4, 15, 16,  5, 18, 19]
    jnt_to      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] 
    for i in range(20):
        if vis[jnt_from[i]] == 0 and vis[jnt_to[i]] == 0:
            paf_vis[:, :, 2*i:2*i + 2] = 0
            continue
        count = np.zeros((self.img_size//self.stride, self.img_size//self.stride))
        centerA = uvd[jnt_to[i], :2]
        centerB = uvd[jnt_from[i], :2]
        vec_map = paf[:, :, 2*i:2*i + 2]
        paf[:, :, 2*i:2*i + 2], count = gen_paf(centerA, centerB,
                                                vec_map, count,
                                                stride=self.stride,
                                                in_size=self.img_size)
    return paf