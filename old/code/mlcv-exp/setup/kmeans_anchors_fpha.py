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

def write_anchors_to_file(centroids, X, anchor_file, img_dim_in_cfg):
    f = open(anchor_file,'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= img_dim_in_cfg[0]/32.
        anchors[i][1] *= img_dim_in_cfg[1]/32.

    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0], anchors[i,1]))

    # There should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:], 0],
                             anchors[sorted_indices[-1:], 1]))

    f.write('Average IOU: %f\n'%(avg_IOU(X, centroids)))
    print()

def kmeans(X, centroids, anchor_file, img_dim_in_cfg):
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
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file,img_dim_in_cfg)
            return

        # Calculate new centroids
        centroid_sums=np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))

        prev_assignments = assignments.copy()
        old_D = D.copy()

def main(argv):
    parent_dir = Path(__file__).absolute().parents[1]
    bbox_set = 'pad10_bbox'
    default_set = str(parent_dir/'data'/'labels'/'fpha_{}_train.txt'.format(bbox_set))

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', default=0, type = int,
                        help='number of clusters\n')
    parser.add_argument('--img_dim', default=416,
                        help='input img dim passed to network')
    parser.add_argument('--bbox_set', default=default_set)
    args = parser.parse_args()

    out_dir = Path(Path(__file__).resolve().parent)/'anchors_fpha_{}'.format(bbox_set)
    if not out_dir.is_dir():
        out_dir.mkdir()

    with open(default_set.replace(bbox_set, 'img'), 'r') as f:
        img_paths = f.read().splitlines()
    with open(parent_dir/'data'/'labels'/'invalid_fpha.txt', 'r') as f:
        invalid_paths = f.readlines()
    invalid_paths = [i.rstrip() for i in invalid_paths]
    bbox = np.loadtxt(args.bbox_set)
    bbox = bbox[:, 2:] # Get width and height only
    bbox = np.asarray([b for i, b in zip(img_paths, bbox) if i not in invalid_paths])
    print(bbox.shape)
    
    img_shape = (args.img_dim, args.img_dim)
    randrange = random.randrange
    if args.num_clusters == 0:
        for num_clusters in range(1, 11):
            anchor_file = Path(out_dir)/'anchors{}.txt'.format(num_clusters)
            indices = [randrange(bbox.shape[0]) for i in range(num_clusters)]
            centroids = bbox[indices]
            kmeans(bbox,centroids, anchor_file, img_shape)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = Path(out_dir)/'anchors{}.txt'.format(args.num_clusters)
        indices = [randrange(bbox.shape[0]) for i in range(args.num_clusters)]
        centroids = bbox[indices]
        print(bbox.shape)
        kmeans(bbox,centroids, anchor_file, img_shape)
        print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main(sys.argv)