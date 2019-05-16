__author__ = 'QiYE'
import numpy
import h5py
import matplotlib.pyplot as plt
import cv2
from src.utils import constants
from math import pi
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
def get_rot(joint_label_uvd,i,j):

    vect = joint_label_uvd[:,i,0:2] - joint_label_uvd[:,j,0:2]#the index is valid for 21joints

    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/pi*180)
    # print numpy.where(rot==180)[0].shape[0]
    rot[numpy.where(rot==180)] =179
    return rot


def rot_img(r0,r1,r2,pred_uvd, gr_uvd ,rotation):

    for i in xrange(0,gr_uvd.shape[0],1):
        M = cv2.getRotationMatrix2D((48,48),-rotation[i],1)
        r0[i] = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in xrange(gr_uvd.shape[1]):
            gr_uvd[i,j,0:2] = numpy.dot(M,numpy.array([gr_uvd[i,j,0]*72+12,gr_uvd[i,j,1]*72+12,1]))/96
        for j in xrange(pred_uvd.shape[1]):
            pred_uvd[i,j,0:2] = numpy.dot(M,numpy.array([pred_uvd[i,j,0]*72+12,pred_uvd[i,j,1]*72+12,1]))/96

        M = cv2.getRotationMatrix2D((24,24),-rotation[i],1)
        r1[i] = cv2.warpAffine(r1[i],M,(48,48),borderValue=1)

        M = cv2.getRotationMatrix2D((12,12),-rotation[i],1)
        r2[i] = cv2.warpAffine(r2[i],M,(24,24),borderValue=1)

    return
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def get_rot_hist(x,bin_size):
    print x.shape
    # Make a normed histogram. It'll be multiplied by 100 later.
    y = plt.hist(x, bins=360/bin_size,range=(-180,180),normed=True)
    print numpy.max(y[0])*100
    print numpy.sum(y[0]*6)
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.xlim(xmin=-180,xmax=180)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.show()


def derot_dataset(dataset,bw_inital_uvd,setname,source_name):

    src_path = '%sdata/%s/source/'%(constants.Data_Path,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    # print f.keys()
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    gr_uvd = f['joint_label_uvd'][...]

    src_path = '%sdata/%s/holi_derot_recur_v2/whl_initial/'%(constants.Data_Path,setname)
    f_derot = h5py.File('%s%s_iter0_whlimg%s.h5'%(src_path,dataset,source_name),'w')
    print r0.shape
    for key in f.keys():
        f.copy(key,f_derot)
    f.close()
    prev_jnt_path ='%s%s%s.npy'%(src_path,dataset,bw_inital_uvd)
    pred_uvd = numpy.load(prev_jnt_path)
    pred_uvd.shape = (pred_uvd.shape[0],21,3)
    upd_rot = get_rot(pred_uvd,0,9)

    gr_rot = get_rot(gr_uvd,0,9)

    rot_err = numpy.mean(numpy.abs(upd_rot-gr_rot))
    print 'rot err', rot_err
    
    get_rot_hist(gr_rot,6)

    f_derot.create_dataset('upd_rot_iter0', data=upd_rot)
    f_derot.create_dataset('pred_uvd', data=pred_uvd)
    f_derot.create_dataset('gr_uvd', data=gr_uvd)

    rot_img(r0,r1,r2,pred_uvd,gr_uvd,upd_rot)

    gr_rot = get_rot(gr_uvd,0,9)
    get_rot_hist(gr_rot,6)


    f_derot.create_dataset('pred_uvd_derot', data=pred_uvd)
    f_derot.create_dataset('gr_uvd_derot', data=gr_uvd)
    f_derot['r0'][...]=r0
    f_derot['r1'][...]=r1
    f_derot['r2'][...]=r2
    f_derot.close()


if __name__ == "__main__":



    # icvl

    derot_dataset(dataset='train',setname='icvl',
                  source_name='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                  bw_inital_uvd='_uvd_whl_r012_21jnts_64_96_128_1_2_adam_lm9')
