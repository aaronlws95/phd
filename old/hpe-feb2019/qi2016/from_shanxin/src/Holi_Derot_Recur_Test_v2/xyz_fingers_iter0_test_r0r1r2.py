__author__ = 'QiYE'
import sys

import theano
import theano.tensor as T
import h5py
import numpy
import scipy.io
import cv2

from load_data import  load_data_finger_jnt_iter0
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import set_params
from src.utils import read_save_format
import File_Name_HoliDR
from src.utils import constants
from src.utils.err_uvd_xyz import err_in_ori_xyz


def test_model(jnt_idx,setname,dataset_path_prefix,source_name,batch_size,patch_size,c1,c2,h1_out_factor,h2_out_factor,model_path):

    model_info='offset_mid%d_r012_21jnts_derot_patch%d'%(jnt_idx,patch_size)
    print model_info, constants.OUT_DIM


    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_finger_jnt_iter0(batch_size,source_name,jnt_idx=jnt_idx,is_shuffle=False,
                                                                                        patch_size=patch_size,patch_pad_width=4,hand_width=96,hand_pad_width=0)

    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches
    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]

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

    save_path =   '%sdata/%s/holi_derot_recur_v2/fingers/best/'%(dataset_path_prefix,setname)
    path = "%sjnt%d_param_cost_offset%s.npy"%(save_path,jnt_idx,model_path)
    set_params(path, model.params)

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


def get_finger_jnt_loc_err(setname,dataset_path_prefix,file_format='mat'):
    jnt_idx_all =[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]
    # jnt_idx_all =[19]
    dataset='test'
    patch_size=40



    if setname =='icvl':
        source_name=File_Name_HoliDR.icvl_source_name
        derot_source_name=File_Name_HoliDR.icvl_iter1_whlimg_derot
        model_path=File_Name_HoliDR.icvl_finger_jnt_model
        xyz_jnt_path=File_Name_HoliDR.icvl_xyz_finger_jnt_save_path
        batch_size =133
        c1=8
        c2=16
        h1_out_factor=1
        h2_out_factor=1
        gr_xyz_path = '%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname)
        gr_roixy_path = '%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname)
    else:
        if setname=='nyu':
            source_name=File_Name_HoliDR.nyu_source_name
            derot_source_name=File_Name_HoliDR.nyu_iter1_whlimg_derot
            model_path=File_Name_HoliDR.nyu_finger_jnt_model
            xyz_jnt_path=File_Name_HoliDR.nyu_xyz_finger_jnt_save_path
            batch_size =4
            c1=24
            c2=48
            h1_out_factor=1
            h2_out_factor=1
            gr_xyz_path = '%sdata/%s/source/%s_%s_xyz_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname)
            gr_roixy_path = '%sdata/%s/source/%s_%s_roixy_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname)
        else:
            if setname =='msrc':
                source_name=File_Name_HoliDR.msrc_source_name
                derot_source_name=File_Name_HoliDR.msrc_iter1_whlimg_derot
                model_path=File_Name_HoliDR.msrc_finger_jnt_model
                xyz_jnt_path=File_Name_HoliDR.msrc_xyz_finger_jnt_save_path
                batch_size =200
                c1=8
                c2=16
                h1_out_factor=1
                h2_out_factor=1
                gr_xyz_path = '%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname)
                gr_roixy_path = '%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname)
            else:
                sys.exit('dataset name shoudle be icvl/nyu/msrc')

    '''don't touch the following part!!!!'''

    f = h5py.File(derot_source_name,'r')
    rot = f['rotation'][...]
    pred_whl_uvd_derot=f['pred_uvd_derot'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    f.close()

    keypoints = scipy.io.loadmat(gr_xyz_path)
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat(gr_roixy_path)
    roixy = keypoints['roixy']


    path = '%sdata/%s/source/%s%s.h5'%(dataset_path_prefix,setname,dataset,source_name)
    f = h5py.File(path,'r')
    uvd_gr = f['joint_label_uvd'][...]
    f.close()


    err_all= []
    for i_idx,jnt_idx in enumerate(jnt_idx_all):
        uvd_offset_norm = test_model(setname=setname,
                                     dataset_path_prefix=dataset_path_prefix,
                    source_name=derot_source_name,
                    model_path=model_path[i_idx],
                    batch_size = batch_size,
                    jnt_idx = jnt_idx,
                    patch_size=patch_size,
                    c1=c1,
                    c2=c2,
                    h1_out_factor=h1_out_factor,
                    h2_out_factor=h2_out_factor)

        predict_uvd=uvd_offset_norm/10+pred_whl_uvd_derot[:,jnt_idx,:]
        """"rot the the norm view to original rotatioin view"""
        for i in xrange(predict_uvd.shape[0]):
            M = cv2.getRotationMatrix2D((48,48),rot[i],1)

            predict_uvd[i,0:2] = (numpy.dot(M,numpy.array([predict_uvd[i,0]*96,predict_uvd[i,1]*96,1]))-12)/72
        xyz,err = err_in_ori_xyz(setname,predict_uvd,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=[jnt_idx])

        err_all.append(err)
        read_save_format.save(xyz_jnt_path[i_idx],data=xyz,format='mat')
    print 'mean jnt err for fingers',numpy.array(err_all).mean()
    print 'jnt err  ', jnt_idx_all, err_all



if __name__ == '__main__':
    """change the NUM_JNTS in src/constants.py to 1"""
    ''''change the path: xyz location of the palm center, file format can be npy or mat'''

    setname='msrc'
    dataset_path_prefix=constants.Data_Path


    get_finger_jnt_loc_err(setname=setname,dataset_path_prefix=dataset_path_prefix,file_format='mat')
