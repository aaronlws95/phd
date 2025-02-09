__author__ = 'Shanxin Yuan'
__date__ = '18 Jan 2017'

import time
import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi
from src.Model.CNN_Model import CNN_Model_multi3_conv3
from src.Model.Train import adam_update,set_adam_params
from src.utils import constants
import error
import scipy.io

def train_model_conv3(setname, cross_test_set, num_test_sample,
                      dataset_path_prefix, model_info, source_name, batch_size,jnt_idx,c1,c2,c3,h1_out_factor,h2_out_factor,lamda,model_name,model_size,bbsize):
    print 'batch_size', batch_size
    # model_info='uvd_whl_r012_21jnts'
    print model_info
    src_path = '%s/'%dataset_path_prefix
    trainortest = 'test'
    path = '%s%s_%s%s.h5'%(src_path,cross_test_set,trainortest,source_name)

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi(batch_size=batch_size,path=path,is_shuffle=False, jnt_idx=jnt_idx)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'num sample after batch', test_set_x0.shape[0]
    print 'n_test_batches', n_test_batches


    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    Y = T.matrix('target')


    model = CNN_Model_multi3_conv3(
            model_info=model_info,
            X0=X0,X1=X1,X2=X2,
            img_size_0 = 96,
            img_size_1=48,
            img_size_2=24,
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

    set_adam_params('%s/%s.npy'%(dataset_path_prefix,model_name),model.params)
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output],on_unused_input='ignore')

    cost_nbatch = 0
    pred_y = numpy.empty_like(test_set_y)
    for minibatch_index in xrange(n_test_batches):
        # print minibatch_index
        slice_idx = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
        x0 = test_set_x0[slice_idx]
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]
        cost_ij,tmp = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
        cost_nbatch+=cost_ij
        pred_y[slice_idx]=tmp

    print 'cost',cost_nbatch/n_test_batches
    pred_y.shape = (pred_y.shape[0],6,3)
    result_save_path = "%s/result_%s"%(dataset_path_prefix,model_size)
    numpy.save('%s_%s_%s_bb%d_pred_norm_uvd_%s'%(result_save_path,cross_test_set,trainortest,bbsize,model_name),pred_y[0:num_test_sample])

    print pred_y.shape
    xyz,xyz_gt,err = error.normuvd2xyz_Shanxin(cross_test_set, jnt_idx, path,pred_y[0:num_test_sample])
    print 'mean err ', numpy.mean(numpy.mean(err))
    print 'mean err for 6 joints', numpy.mean(err,axis=0)
    scipy.io.savemat('%s%s_%s_bb%d_xyz_%s'%(result_save_path,cross_test_set,trainortest,bbsize,model_name),
                     {'xyz_pred':xyz,'xyz_jnt_gt':xyz_gt})




if __name__ == '__main__':

    bbsize=200
    print 'bbsize=',bbsize
    setname = 'icvl'
    dataset_path_prefix = '/media/Data/shanxin/StructuralHand/Mega/data/%s' % setname
    sourcename = '_norm_hand_uvd_rootmid'
    # batch_size=256
    batch_size=128

    model_info='uvd_PALM_r012_6_jnts'
    Param_name = 'param_cost_%s_64_96_128_1_2_adam_lm99' % model_info
    train_model_conv3(setname=setname,
                    cross_test_set = setname,
                    num_test_sample = 1596,
                    dataset_path_prefix=dataset_path_prefix,
                    model_info = model_info,
                    model_name=Param_name,
                    model_size='PALM',
                    bbsize=bbsize,
                    source_name=sourcename,
                    lamda=0.0001,
                    batch_size = batch_size,
                    jnt_idx = range(0,6,1),
                    c1=64,
                    c2=96,
                    c3=128,
                    h1_out_factor=1,
                    h2_out_factor=2)

