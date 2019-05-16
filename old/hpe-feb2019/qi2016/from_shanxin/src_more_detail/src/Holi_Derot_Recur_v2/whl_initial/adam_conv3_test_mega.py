__author__ = 'QiYE'
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
def train_model_conv3(setname, cross_test_set,num_test_sample,
                      dataset_path_prefix,source_name,batch_size,jnt_idx,c1,c2,c3,h1_out_factor,h2_out_factor,
                      bbsize,model_size,model_name):
    print 'batch_size', batch_size
    model_info='uvd_whl_r012_21jnts'
    print model_info
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)

    dataset = 'test'
    path = '%s%s_%s%s.h5'%(src_path,cross_test_set,dataset,source_name)

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi(batch_size=batch_size,path=path,is_shuffle=False, jnt_idx=jnt_idx)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'num sample after batch', test_set_x0.shape[0]
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
    #
    # model_name= 'uvd_whl_r012_21jnts_64_96_128_1_2_adam_lm99'
    # set_adam_params('%sdata/%s/whole/param_cost_%s.npy'%(dataset_path_prefix,setname,model_name),model.params)
    #
    # model_name= 'uvd_whl_r012_21jnts_half_64_96_128_1_2_adam_lm300_ep22_000379'
    # set_adam_params('%sdata/%s/whole/inter/param_cost_%s.npy'%(dataset_path_prefix,setname,model_name),model.params)


    set_adam_params('%sdata/%s/whole/best/param_cost_%s.npy'%(dataset_path_prefix,setname,model_name),model.params)

    print model_name,
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
    pred_y.shape = (pred_y.shape[0],21,3)
    result_save_path = "%sdata/%s/whole/result_%s"%(dataset_path_prefix,setname,model_size)
    # result_save_path = "%sdata/%s/whole/result_yang"%(dataset_path_prefix,setname)
    numpy.save('%s/%s_%s_pred_norm_uvd_bb%d_%s'%(result_save_path,cross_test_set,dataset,bbsize,model_name),pred_y[0:num_test_sample])

    xyz,xyz_jnt_gt = error.normuvd2xyz_mega(cross_test_set,path,pred_y[0:num_test_sample])
    # numpy.save('%s/%s_%s_xyz_%s'%(result_save_path,cross_test_set,dataset,model_name),xyz)
    scipy.io.savemat('%s/%s_%s_xyz_bb%d_%s'%(result_save_path,cross_test_set,dataset,bbsize,model_name),
                     {'xyz_pred':xyz,'xyz_jnt_gt':xyz_jnt_gt})


if __name__ == '__main__':


    # dataset_path_prefix=constants.Data_Path
    # setname='mega'
    # cross_test_set = 'mega'
    # dataset='test'
    # model_name= 'uvd_whl_r012_21jnts_64_96_128_1_2_adam_lm99'
    # result_save_path = "%sdata/%s/whole/result"%(dataset_path_prefix,setname)
    # pred_y = numpy.load('%s/%s_%s_half_pred_norm_uvd_%s.npy'%(result_save_path,cross_test_set,dataset,model_name))
    #
    # dataset = 'test'
    # src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    # path = '%s%s_%s_norm_hand_uvd_rootmid.h5'%(src_path,cross_test_set,dataset)
    # xyz,xyz_jnt_gt = error.normuvd2xyz_mega(setname=cross_test_set,path=path,cur_norm_uvd=pred_y)
    # scipy.io.savemat('%s/%s_%s_hlaf_xyz_%s'%(result_save_path,cross_test_set,dataset,model_name),
    #                  {'xyz_pred':xyz,'xyz_jnt_gt':xyz_jnt_gt})
    # # numpy.save('%s/%s_%s_xyz_%s'%(result_save_path,cross_test_set,dataset,model_name),[xyz,xyz_jnt_gt])

    #
    #
    # uvd_whl_r012_21jnts_16_64_96_128_1_2_adam_lm300_best

    model_size = ['half','quater','8','16']
    for ms in model_size:
        train_model_conv3(setname='mega',
                          cross_test_set = 'mega',
                          model_name= 'uvd_whl_r012_21jnts_%s_64_96_128_1_2_adam_lm300_best'%ms,
                          model_size=ms,
                          bbsize=260,
                          num_test_sample = 91658,
                         dataset_path_prefix=constants.Data_Path,
                source_name='_norm_hand_uvd_rootmid_half',
                    batch_size = 1024,
                    jnt_idx = range(0,21,1),
                    c1=64,
                    c2=96,
                    c3=128,
                    h1_out_factor=1,
                    h2_out_factor=2)

    train_model_conv3(setname='mega',
                      cross_test_set = 'mega',
                      model_name= 'uvd_whl_r012_21jnts_64_96_128_1_2_adam_lm99',
                      model_size='full',
                      bbsize=260,
                      num_test_sample = 91658,
                     dataset_path_prefix=constants.Data_Path,
            source_name='_norm_hand_uvd_rootmid_half',
                batch_size = 1024,
                jnt_idx = range(0,21,1),
                c1=64,
                c2=96,
                c3=128,
                h1_out_factor=1,
                h2_out_factor=2)
    #
    # model_size = ['half','quater','8','16']
    # for ms in model_size:
    #     train_model_conv3(setname='mega',
    #                       cross_test_set = 'mega',
    #                       model_name= 'uvd_whl_r012_21jnts_%s_64_96_128_1_2_adam_lm300_best'%ms,
    #                       model_size=ms,
    #                       bbsize=260,
    #                       num_test_sample = 37000,
    #                      dataset_path_prefix=constants.Data_Path,
    #             source_name='_norm_hand_uvd_rootmid_yang',
    #                 batch_size = 1024,
    #                 jnt_idx = range(0,21,1),
    #                 c1=64,
    #                 c2=96,
    #                 c3=128,
    #                 h1_out_factor=1,
    #                 h2_out_factor=2)
    # train_model_conv3(setname='mega',
    #                   cross_test_set = 'mega',
    #                   model_name= 'uvd_whl_r012_21jnts_64_96_128_1_2_adam_lm99',
    #                   model_size='',
    #                   bbsize=260,
    #                   num_test_sample = 37000,
    #                  dataset_path_prefix=constants.Data_Path,
    #         source_name='_norm_hand_uvd_rootmid_yang',
    #             batch_size = 1024,
    #             jnt_idx = range(0,21,1),
    #             c1=64,
    #             c2=96,
    #             c3=128,
    #             h1_out_factor=1,
    #             h2_out_factor=2)