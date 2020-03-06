__author__ = 'QiYE'
        # print i

from src.utils import constants

__author__ = 'QiYE'
import numpy
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation  as interplt
import h5py

for i in xrange(100,2000,5):
    filename_prefix = "%07d" % (i+1)
    dataset_dir='D:/Project/3DHandPose/Data_3DHandPoseDataset/NYU_dataset/NYU_dataset/test/'
    depth = Image.open('%sdepth_1_%s.png' % (dataset_dir, filename_prefix))

    # dataset_dir='D:/Project/3DHandPose/Data_3DHandPoseDataset/ICVL_dataset_v2_msrc_format/Test/depth/'
    # depth = Image.open('%sdepth_1_%s.png' % (dataset_dir, filename_prefix))
    depth = numpy.asarray(depth, dtype='uint16')
    depth = numpy.asarray(depth, dtype='uint16')
    depth = depth[:, :, 2]+numpy.left_shift(depth[:, :, 1], 8)

    """d1 d2 are the left top axes of the hand area to be extracted"""
    d1 = depth.shape[0]
    d2 = depth.shape[1]
    plt.imshow(depth[:, :], cmap='gray')

    plt.show()
