__author__ = 'QiYE'
import sys

import theano
import theano.tensor as T
import h5py
import numpy
import scipy.io

from load_data import  load_data_finger_jnt
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import set_params
from src.utils.err_uvd_xyz import uvd_to_xyz_error_single_v2,xyz_to_uvd_derot
from src.utils import read_save_format
import File_Name_HDR
from src.utils import constants


def test_model(setname,dataset_path_prefix,source_name,prev_jnt_uvd_derot,
               batch_size,jnt_idx,patch_size,offset_depth_range,c1,c2,h1_out_factor,h2_out_factor,model_path):

    print 'offset_depth_range ',offset_depth_range
    model_info='offset_mid%d_r012_21jnts_derotpatch%d'%(jnt_idx[0],patch_size)


    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_finger_jnt(path=source_name,prev_jnt_uvd_pred=prev_jnt_uvd_derot,
                                                                   jnt_idx=jnt_idx,is_shuffle=False,
                                                                                        patch_size=patch_size,
                                                                                        patch_pad_width=4,hand_width=96,hand_pad_width=0,
                                                                                        offset_depth_range=offset_depth_range)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches
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
                )
    cost = model.cost(Y)

    save_path =   '%sdata/%s/hier_derot_recur_v2/mid/best/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_path)
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



def get_mid_loc_err(setname,dataset_path_prefix,file_format):
    jnt_idx_all5 = [[2],[6],[10],[14],[18]]
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
        model_path=File_Name_HDR.icvl_mid_model
        xyz_jnt_path=File_Name_HDR.icvl_xyz_mid_jnt_save_path
        prev_jnt_path=File_Name_HDR.icvl_xyz_bw_jnt_save_path
        batch_size =133
    else:
        if setname=='nyu':
            source_name=File_Name_HDR.nyu_source_name
            derot_source_name=File_Name_HDR.nyu_iter1_whlimg_derot
            model_path=File_Name_HDR.nyu_mid_model
            xyz_jnt_path=File_Name_HDR.nyu_xyz_mid_jnt_save_path
            prev_jnt_path=File_Name_HDR.nyu_xyz_bw_jnt_save_path
            batch_size =100
        else:
            if setname =='msrc':
                source_name=File_Name_HDR.msrc_source_name
                derot_source_name=File_Name_HDR.msrc_iter1_whlimg_derot
                model_path=File_Name_HDR.msrc_mid_model
                xyz_jnt_path=File_Name_HDR.msrc_xyz_min_jnt_save_path
                prev_jnt_path=File_Name_HDR.msrc_xyz_bw_jnt_save_path
                batch_size =100
            else:
                sys.exit('dataset name shoudle be icvl/nyu/msrc')

    '''don't touch the following part!!!!'''

    f = h5py.File(derot_source_name,'r')
    base_wrist_pred=f['pred_bw_uvd_derot'][...]
    rot = f['rotation'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    # derot_uvd = f['joint_label_uvd'][...]
    f.close()

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']


    path = '%sdata/%s/source/%s%s.h5'%(dataset_path_prefix,setname,dataset,source_name)
    f = h5py.File(path,'r')
    uvd_gr = f['joint_label_uvd'][...]
    f.close()


    err_all5 = []
    for i,jnt_idx in enumerate(jnt_idx_all5):
        print prev_jnt_path[i]
        prev_jnt_xyz=read_save_format.load(prev_jnt_path[i+1],format=file_format)
        print  prev_jnt_xyz.shape
        prev_jnt_uvd_derot = xyz_to_uvd_derot(prev_jnt_xyz,setname=setname,rot=rot,jnt_idx=jnt_idx,
                                              roixy=roixy,rect_d1d2w=rect_d1d2w,depth_dmin_dmax=depth_dmin_dmax,orig_pad_border=orig_pad_border)


        uvd_offset_norm = test_model(setname=setname,
                                     dataset_path_prefix=dataset_path_prefix,
                    source_name=derot_source_name,
                    model_path=model_path[i],
                    batch_size = batch_size,
                    jnt_idx = jnt_idx,
                    patch_size=patch_size,
                    offset_depth_range=offset_depth_range,
                    c1=c1,c2=c2,
                    h1_out_factor=h1_out_factor,
                    h2_out_factor=h2_out_factor,
                    prev_jnt_uvd_derot=prev_jnt_uvd_derot
            )

        xyz,err = uvd_to_xyz_error_single_v2(setname=setname,uvd_pred_offset=uvd_offset_norm,rot=rot,
                               prev_jnt_uvd_derot=prev_jnt_uvd_derot,patch_size=patch_size,jnt_idx =jnt_idx,offset_depth_range=offset_depth_range,
                               uvd_gr=uvd_gr,xyz_true=xyz_true,
                               roixy=roixy,rect_d1d2w=rect_d1d2w,depth_dmin_dmax=depth_dmin_dmax,orig_pad_border=orig_pad_border)
        err_all5.append(err)
        read_save_format.save(xyz_jnt_path[i],data=xyz,format='mat')

    print 'jnt err for mid ', jnt_idx_all5, err_all5
    print 'mean jnt err for mid',numpy.array(err_all5).mean()


if __name__ == '__main__':
    """change the NUM_JNTS in src/constants.py to 1"""
    ''''change the path: xyz location of the palm center, file format can be npy or mat'''

    setname='icvl'
    dataset_path_prefix=constants.Data_Path
    file_format='mat'

    get_mid_loc_err(setname=setname,dataset_path_prefix=dataset_path_prefix,file_format=file_format)
