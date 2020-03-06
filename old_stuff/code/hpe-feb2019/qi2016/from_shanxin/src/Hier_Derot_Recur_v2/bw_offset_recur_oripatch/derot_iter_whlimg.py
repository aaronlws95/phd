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
    print numpy.where(rot==180)[0].shape[0]
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
    # print x.shape
    # Make a normed histogram. It'll be multiplied by 100 later.
    y = plt.hist(x, bins=360/bin_size,range=(-180,180),normed=True)
    # print numpy.max(y[0])*100
    # print numpy.sum(y[0]*6)
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.xlim(xmin=-180,xmax=180)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.show()

def derot_dataset(dataset,setname,source_name):

    src_path = '%sdata/%s/source/'%(constants.Data_Path,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    # print f.keys()
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    gr_uvd = f['joint_label_uvd'][...]
    src_path = '%sdata/%s/hier_derot_recur_v2/bw_offset/best/'%(constants.Data_Path,setname)
    f_derot = h5py.File('%s%s_iter1_whlimg%s.h5'%(src_path,dataset,source_name),'w')
    print r0.shape
    for key in f.keys():
        f.copy(key,f_derot)
    f.close()


    direct = '%sdata/%s/hier_derot_recur_v2/final_xyz_uvd/'%(constants.Data_Path,setname)
    pred_bw_uvd = numpy.empty((r0.shape[0],6,3),dtype='float32')
    for i in xrange(6):
        pred_bw_uvd[:, i, :] = numpy.load('%s%s_absuvd%s.npy'%(direct,dataset,param_names[i]))

    upd_rot = get_rot(pred_bw_uvd,0,3)
    get_rot_hist(upd_rot,6)
    gr_rot = get_rot(gr_uvd,0,9)

    rot_err = numpy.mean(numpy.abs(upd_rot-gr_rot))
    print 'rot err', rot_err
    
    get_rot_hist(gr_rot,6)

    rot_img(r0,r1,r2,pred_bw_uvd,gr_uvd,upd_rot)

    gr_rot = get_rot(gr_uvd,0,9)
    get_rot_hist(gr_rot,6)

    f_derot.create_dataset('upd_rot', data=upd_rot)
    f_derot.create_dataset('gr_uvd_derot', data=gr_uvd)
    f_derot.create_dataset('pred_bw_uvd_derot', data=pred_bw_uvd)
    f_derot['r0'][...]=r0
    f_derot['r1'][...]=r1
    f_derot['r2'][...]=r2
    f_derot.close()


if __name__ == "__main__":

    # read_derot_dataset(dataset='train',setname='icvl',file_name='_iter1_whlimg_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200')
    # derot_dataset(dataset='train',setname='nyu',
    #               source_name='_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300')


    # icvl
    # upd_pred_name=['_absuvd_bw0_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep465',
    #           '_absuvd_bw1_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep285',
    #            '_absuvd_bw2_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep300',
    #            '_absuvd_bw3_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep715',
    #            '_absuvd_bw4_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep675',
    #            '_absuvd_bw5_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep415'
    #            ]
    # derot_dataset(dataset='test',setname='icvl',
    #               source_name='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200')

    # nyu
    param_names=['_egoff_iter1_bw0_beta0_24_48_1_1_adam_lm300',
                 '_egoff_iter1_bw1_beta0_24_48_1_1_adam_lm300',
                 '_egoff_iter1_bw5_beta0_24_48_1_1_adam_lm300',
                 '_egoff_iter1_bw9_beta0_24_48_1_1_adam_lm300',
                 '_egoff_iter1_bw13_beta0_24_48_1_1_adam_lm300',
                 '_egoff_iter1_bw17_r012_24_48_1_1_adam_lm300']
    derot_dataset(dataset='train',setname='icvl',
                  source_name='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200')

    #
    # derot_dataset(dataset='test',setname='nyu',
    #               source_name='_nyu_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300')


    # # icvl
    # upd_pred_name=['_absuvd_bw0_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep465',
    #           '_absuvd_bw1_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep285',
    #            '_absuvd_bw2_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep300',
    #            '_absuvd_bw3_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep715',
    #            '_absuvd_bw4_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep675',
    #            '_absuvd_bw5_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep415'
    #            ]
    # derot_dataset(dataset='test',setname='icvl',
    #               source_name='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200')


    # # msrc
    # upd_pred_name=['_absuvd_bw0_r012_egoff_c0064_h11_h22_gm0_lm3000_yt0_ep155',
    #           '_absuvd_bw1_r012_egoff_c0064_h11_h22_gm0_lm3000_yt0_ep170',
    #            '_absuvd_bw2_r012_egoff_c0064_h11_h22_gm0_lm3000_yt0_ep105',
    #            '_absuvd_bw3_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep75',
    #            '_absuvd_bw4_r012_egoff_c0064_h11_h22_gm0_lm3000_yt0_ep200',
    #            '_absuvd_bw5_r012_egoff2_c0064_h11_h22_gm0_lm3000_yt0_ep185'
    #            ]
    # derot_dataset(dataset='test',setname='msrc',
    #               source_name='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300')