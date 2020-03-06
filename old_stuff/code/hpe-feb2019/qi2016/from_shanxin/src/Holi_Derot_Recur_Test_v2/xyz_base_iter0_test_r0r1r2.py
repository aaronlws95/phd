__author__ = 'QiYE'

import sys
from math import pi

import theano
import theano.tensor as T
import h5py
import numpy
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from matplotlib.ticker import FuncFormatter

from load_data import  load_data_iter0
from src.Model.CNN_Model import CNN_Model_multi3_conv3
from src.Model.Train import set_params
from src.utils import constants
import File_Name_HoliDR
from src.utils.err_uvd_xyz import err_in_ori_xyz


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

def derot_dataset(setname,source_name,derot_source_name,pred_uvd):

    path = '%sdata/%s/source/test%s.h5'%(constants.Data_Path,setname,source_name)
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    gr_uvd = f['joint_label_uvd'][...]
    f_derot = h5py.File(derot_source_name,'w')
    print r0.shape
    for key in f.keys():
        f.copy(key,f_derot)
    f.close()

    upd_rot = get_rot(pred_uvd,0,9)
    gr_rot = get_rot(gr_uvd,0,9)

    rot_err = numpy.mean(numpy.abs(upd_rot-gr_rot))
    print 'rot err', rot_err

    rot_img(r0,r1,r2,pred_uvd,gr_uvd,upd_rot)

    f_derot.create_dataset('rotation', data=upd_rot)
    f_derot.create_dataset('gr_uvd_derot', data=gr_uvd)
    f_derot.create_dataset('pred_uvd_derot', data=pred_uvd)
    f_derot['r0'][...]=r0
    f_derot['r1'][...]=r1
    f_derot['r2'][...]=r2
    f_derot.close()
    print 'derot whl image saved', derot_source_name

def test_model_conv3(dataset,setname,dataset_path_prefix, source_name,batch_size,jnt_idx,
                     c1,c2,c3,h1_out_factor,h2_out_factor,model_path):

    model_info='uvd_bw_r012_21jnts'
    print model_info
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)

    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_iter0(batch_size,path,is_shuffle=False, jnt_idx=jnt_idx)

    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches



    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')



    model = CNN_Model_multi3_conv3(
        model_info=model_info,
        X0=X0,X1=X1,X2=X2,
              img_size_0 = img_size_0,
              img_size_1=img_size_1,
              img_size_2=img_size_2,
              is_train=is_train,
                c00= c1,
                kernel_c00= 5,
                pool_c00= 4,
                c01= c2,
                kernel_c01= 4,
                pool_c01= 2 ,
                c02= c3,
                kernel_c02= 3,
                pool_c02= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 3,
                pool_c11= 2 ,
                c12= c3,
                kernel_c12= 3,
                pool_c12= 2 ,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 5,
                pool_c21= 1 ,
                c22= c3,
                kernel_c22= 3,
                pool_c22= 1 ,

                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.cost(Y)

    save_path =   '%sdata/%s/holi_derot_recur_v2/whl_initial/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_path)
    print model_save_path
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output], on_unused_input='ignore')

    cost_nbatch = 0
    uvd_norm = numpy.empty_like(test_set_y)

    for minibatch_index in xrange(n_test_batches):
        slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
        x0 = test_set_x0[slice_idx]
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]

        cost_ij, uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
        uvd_norm[slice_idx] = uvd_batch
        cost_nbatch+=cost_ij

    return uvd_norm



def get_bw_initial_xyz_err(setname,dataset_path_prefix):

    ''''change the path: xyz location of the palm center, file format can be npy or mat'''

    dataset='test'
    c1=64
    c2=96
    c3=128
    h1_out_factor=1
    h2_out_factor=1
    if setname =='icvl':
        source_name=File_Name_HoliDR.icvl_source_name
        derot_source_name=File_Name_HoliDR.icvl_iter0_whlimg_derot
        model_path=File_Name_HoliDR.icvl_initial_model
        batch_size =133
        gr_xyz_path = '%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname)
        gr_roixy_path = '%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname)
    else:
        if setname=='nyu':
            source_name=File_Name_HoliDR.nyu_source_name
            derot_source_name=File_Name_HoliDR.nyu_iter0_whlimg_derot
            model_path=File_Name_HoliDR.nyu_initial_model
            batch_size =4
            gr_xyz_path = '%sdata/%s/source/%s_%s_xyz_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname)
            gr_roixy_path = '%sdata/%s/source/%s_%s_roixy_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname)
        else:
            if setname =='msrc':
                source_name=File_Name_HoliDR.msrc_source_name
                derot_source_name=File_Name_HoliDR.msrc_iter1_whlimg_derot
                model_path=File_Name_HoliDR.msrc_initial_model
                batch_size =200
                gr_xyz_path = '%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname)
                gr_roixy_path = '%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname)
            else:
                sys.exit('dataset name shoudle be icvl/nyu/msrc')



    keypoints = scipy.io.loadmat(gr_xyz_path)
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat(gr_roixy_path)
    roixy = keypoints['roixy']

    path = '%sdata/%s/source/%s%s.h5'%(dataset_path_prefix,setname,dataset,source_name)
    f = h5py.File(path,'r')
    uvd_gr = f['joint_label_uvd'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    f.close()


    jnt_idx =range(0,21,1)
    pred_uvd= test_model_conv3(dataset=dataset,setname=setname,dataset_path_prefix=dataset_path_prefix,
                               source_name=source_name,batch_size=batch_size,jnt_idx=jnt_idx,
                     c1=c1,c2=c2,c3=c3,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,model_path=model_path)

    err_ori_xyz = err_in_ori_xyz(setname,pred_uvd.reshape(pred_uvd.shape[0],21,3),uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)

    derot_dataset(setname=setname,source_name=source_name,derot_source_name=derot_source_name,pred_uvd=pred_uvd.reshape(pred_uvd.shape[0],21,3))

if __name__ == '__main__':
    '''the first iteration for bw refinement and save the xyz locations specified in xyz_jnt_path'''
    """change the NUM_JNTS in src/constants.py to 1"""

    # setname='icvl'
    # setname='nyu'
    setname='msrc'
    dataset_path_prefix=constants.Data_Path
    get_bw_initial_xyz_err(dataset_path_prefix=constants.Data_Path, setname=setname)