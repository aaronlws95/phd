# https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py
import argparse
import numpy as np
import sys
import shutil
import random
import math
from pathlib import Path

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src import ROOT

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w  >=  w and c_h  >=  h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w - w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h - h))
        else: # Both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # (k,) shape
    return np.array(similarities)

def avg_IOU(X,centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        sum += max(IOU(X[i], centroids))
    return sum/n

def print_anchors(centroids, X, img_rsz, net_stride):
    anchors = centroids.copy()

    for i in range(anchors.shape[0]):
        anchors[i][0] *= img_rsz/net_stride
        anchors[i][1] *= img_rsz/net_stride

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices:
        print('%0.2f, %0.2f, '%(anchors[i, 0], anchors[i, 1]), end='')
    print('\n')

    print('Average IOU: %f\n'%(avg_IOU(X, centroids)))

def kmeans(X, centroids, img_rsz, net_stride):
    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))

        # Assign samples to centroids
        assignments = np.argmin(D,axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            print_anchors(centroids, X, img_rsz, net_stride)
            return

        # Calculate new centroids
        centroid_sums=np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))

        prev_assignments = assignments.copy()
        old_D = D.copy()

def main():
    parent_dir      = Path(__file__).absolute().parents[1]
    bbox_path       = 'concatdata_yolov2_bbox_exp1_bbox_norm_train.txt'
    bbox_path       = str(parent_dir/'data'/'labels'/bbox_path)
    img_rsz         = 416
    num_clusters    = 5
    net_stride      = 32

    bbox = np.loadtxt(bbox_path)
    bbox = bbox[:, 2:] # Get width and height only

    randrange = random.randrange
    indices = [randrange(bbox.shape[0]) for i in range(num_clusters)]
    centroids = bbox[indices]
    kmeans(bbox, centroids, img_rsz, net_stride)
    print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main()