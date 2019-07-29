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

from load_data import  load_data_iter1
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import set_params
from src.utils import read_save_format,constants
import File_Name_HDR
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

    upd_rot = get_rot(pred_uvd,0,3)
    gr_rot = get_rot(gr_uvd,0,9)

    rot_err = numpy.mean(numpy.abs(upd_rot-gr_rot))
    print 'rot err', rot_err

    rot_img(r0,r1,r2,pred_uvd,gr_uvd,upd_rot)

    f_derot.create_dataset('rotation', data=upd_rot)
    f_derot.create_dataset('gr_uvd_derot', data=gr_uvd)
    f_derot.create_dataset('pred_bw_uvd_derot', data=pred_uvd)
    f_derot['r0'][...]=r0
    f_derot['r1'][...]=r1
    f_derot['r2'][...]=r2
    f_derot.close()
    print 'derot whl image saved', derot_source_name


def train_model_multi3(dataset,setname, dataset_path_prefix,source_name,batch_size,jnt_idx,patch_size,c1,c2,h1_out_factor,h2_out_factor,model_save_path):

    model_info='uvd_bw%s_r012_egoff'%jnt_idx[0]


    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_iter1(batch_size,path=source_name,is_shuffle=False,jnt_idx=jnt_idx,
                                                                                         patch_size=patch_size,patch_pad_width=4,hand_width=96,hand_pad_width=0)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    print 'n_test_batches', n_test_batches


    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')

    model = CNN_Model_multi3(X0=X0,X1=X1,X2=X2,
                             model_info=model_info,
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

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 3,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 1,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 2 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)
    cost = model.cost(Y)

    save_path = '%sdata/%s/hier_derot_recur_v2/bw_offset/best/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_save_path)
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
    print 'cost', cost_nbatch/n_test_batches
    return uvd_norm



def get_bw_xyz_err(setname,dataset_path_prefix):

    ''''change the path: xyz location of the palm center, file format can be npy or mat'''

    dataset='test'
    patch_size=40
    offset_depth_range=0.4
    c1=24
    c2=48
    h1_out_factor=1
    h2_out_factor=1
    if setname =='icvl':
        source_name=File_Name_HDR.icvl_source_name
        derot_source_name=File_Name_HDR.icvl_iter1_whlimg_derot
        model_path=File_Name_HDR.icvl_bw_model
        xyz_jnt_path=File_Name_HDR.icvl_xyz_bw_jnt_save_path
        bw_initial_patch = File_Name_HDR.icvl_iter0_whlimg_derot
        batch_size =133
    else:
        if setname=='nyu':
            source_name=File_Name_HDR.nyu_source_name
            derot_source_name=File_Name_HDR.nyu_iter1_whlimg_derot
            model_path=File_Name_HDR.nyu_bw_model
            xyz_jnt_path=File_Name_HDR.nyu_xyz_bw_jnt_save_path
            bw_initial_patch = File_Name_HDR.nyu_bw_initial_patch
            batch_size =100
        else:
            if setname =='msrc':
                source_name=File_Name_HDR.msrc_source_name
                derot_source_name=File_Name_HDR.msrc_iter1_whlimg_derot
                model_path=File_Name_HDR.msrc_bw_model
                xyz_jnt_path=File_Name_HDR.msrc_xyz_bw_jnt_save_path
                bw_initial_patch = File_Name_HDR.msrc_bw_initial_patch
                batch_size =100
            else:
                sys.exit('dataset name shoudle be icvl/nyu/msrc')
    print bw_initial_patch
    f = h5py.File(bw_initial_patch,'r')
    pred_bw_uvd_derot_0 = f['pred_bw_uvd_derot'][...]
    rot = f['rotation'][...]
    f.close()
    pred_bw_uvd_derot_0.shape =(pred_bw_uvd_derot_0.shape[0],6,3)


    path = '%sdata/%s/source/%s%s.h5'%(dataset_path_prefix,setname,dataset,source_name)
    f = h5py.File(path,'r')
    uvd_gr = f['joint_label_uvd'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    f.close()

    uvd_offset_norm=numpy.empty_like(pred_bw_uvd_derot_0)
    jnt_idx = [0,1,5,9 ,13,17]
    for i,idx in enumerate(jnt_idx):
        uvd_offset_norm[:,i,:] =train_model_multi3(dataset=dataset,setname=setname, dataset_path_prefix=dataset_path_prefix,
                                                   source_name=bw_initial_patch,batch_size=batch_size,
                                                   jnt_idx=[idx],patch_size=patch_size,
                                                   c1=c1,c2=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,
                                                   model_save_path=model_path[i])



    predict_bw_uvd_1=uvd_offset_norm/10+pred_bw_uvd_derot_0
    """"rot the the norm view to original rotatioin view"""
    for i in xrange(predict_bw_uvd_1.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot[i],1)
        for j in xrange(predict_bw_uvd_1.shape[1]):
            predict_bw_uvd_1[i,j,0:2] = (numpy.dot(M,numpy.array([predict_bw_uvd_1[i,j,0]*96,predict_bw_uvd_1[i,j,1]*96,1]))-12)/72

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']
    xyz,err = err_in_ori_xyz(setname,predict_bw_uvd_1,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)


    print xyz.shape
    read_save_format.save(xyz_jnt_path[0],data=xyz[:,0,:],format='mat')
    read_save_format.save(xyz_jnt_path[1],data=xyz[:,1,:],format='mat')
    read_save_format.save(xyz_jnt_path[2],data=xyz[:,2,:],format='mat')
    read_save_format.save(xyz_jnt_path[3],data=xyz[:,3,:],format='mat')
    read_save_format.save(xyz_jnt_path[4],data=xyz[:,4,:],format='mat')
    read_save_format.save(xyz_jnt_path[5],data=xyz[:,5,:],format='mat')

    derot_dataset(setname=setname,source_name=source_name,derot_source_name=derot_source_name,pred_uvd=predict_bw_uvd_1)

if __name__ == '__main__':
    '''the first iteration for bw refinement and save the xyz locations specified in xyz_jnt_path'''
    """change the NUM_JNTS in src/constants.py to 1"""

    setname='icvl'
    # setname='nyu'
    # setname='msrc'
    dataset_path_prefix=constants.Data_Path
    get_bw_xyz_err(dataset_path_prefix=constants.Data_Path, setname=setname)