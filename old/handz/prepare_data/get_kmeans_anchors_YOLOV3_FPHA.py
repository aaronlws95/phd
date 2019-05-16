# https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py
import os
import argparse
import numpy as np
import sys
import os
import shutil
import random 
import math

sys.path.append(os.path.dirname(os.path.abspath("")))
from utils.lmdb_utils import  *
import utils.YOLO_utils as YOLO

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file,img_dim_in_cfg):
    f = open(anchor_file,'w')
    
    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0]*=img_dim_in_cfg[0]
        anchors[i][1]*=img_dim_in_cfg[1]
         
    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])
        
    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    #there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))
    
    f.write('Average IOU: %f\n'%(avg_IOU(X,centroids)))
    print()

def kmeans(X,centroids,anchor_file,img_dim_in_cfg):
    
    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = []
        iter+=1
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))

        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        
        if (assignments == prev_assignments).all():
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file,img_dim_in_cfg)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', default = 0, type = int, 
                        help='number of clusters\n')  
    args = parser.parse_args()
    
    output_dir = 'anchors_fpha_yolov3'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_prefix = 'train_fpha_root'
    keys = get_keys(os.path.join(YOLO.FPHA_DIR, save_prefix + '_keys_cache.p'))
    dataroot = os.path.join(YOLO.FPHA_DIR, save_prefix + '_bbox.lmdb')
    bbox = read_all_lmdb_dataroot(keys, dataroot, 'float32', 4)
    bbox = bbox[:, 2:] # get width and height only
    
    img_dim_in_cfg = (416, 416)
    
    if args.num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = os.path.join(output_dir,'anchors%d.txt'%(num_clusters))

            indices = [random.randrange(bbox.shape[0]) for i in range(num_clusters)]
            centroids = bbox[indices]
            kmeans(bbox,centroids,anchor_file,img_dim_in_cfg)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = os.path.join(output_dir,'anchors%d.txt'%(args.num_clusters))
        indices = [ random.randrange(bbox.shape[0]) for i in range(args.num_clusters)]
        centroids = bbox[indices]
        print(bbox.shape)
        kmeans(bbox,centroids,anchor_file,img_dim_in_cfg)
        print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main(sys.argv)    