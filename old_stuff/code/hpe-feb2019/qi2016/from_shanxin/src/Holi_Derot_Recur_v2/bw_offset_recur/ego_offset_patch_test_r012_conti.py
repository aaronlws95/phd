__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_r012_ego_offset
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import update_params,get_gradients,update_params2,set_params
import time
from src.utils import constants
def train_model(num_iter,dataset,setname, dataset_path_prefix,source_name,batch_size,jnt_idx,patch_size,c1,c2,h1_out_factor,h2_out_factor,model_save_path,offset_save_path):
    if num_iter == 1:
        model_info='egoff_iter1_bw%s_r012'%jnt_idx[0]
        src_path = '%sdata/%s/holi_derot_recur_v2/whl_initial/'%(dataset_path_prefix,setname)
    else:
        model_info='egoff_iter2_bw%s_r012'%jnt_idx[0]
        src_path = '%sdata/%s/holi_derot_recur_v2/bw_offset/best/'%(dataset_path_prefix,setname)

    path = '%s%s%s.h5'%(src_path,dataset,source_name)

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_r012_ego_offset(batch_size,path=path,is_shuffle=False,jnt_idx=jnt_idx,
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

    save_path =   '%sdata/%s/holi_derot_recur_v2/bw_offset/'%(dataset_path_prefix,setname)
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

    numpy.save("%s%s%s.npy"%(save_path,dataset,offset_save_path),uvd_norm)


if __name__ == '__main__':

    # train_model(dataset='train',
    #             setname='icvl',
    #             dataset_path_prefix=constants.Data_Path,
    #             source_name='_iter0_whlimg_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             batch_size = 8,
    #             jnt_idx = [9],
    #             c1=24,
    #             c2=48,
    #             patch_size=40,
    #             h1_out_factor=1,
    #             h2_out_factor=1,
    #             model_save_path = 'param_cost_egoff_adam_iter1_bw9_r012_24_48_1_1_adam_lm29',
    #             offset_save_path ='_egoff_adam_iter1_bw9_r012_24_48_1_1_adam_lm29' )

    train_model(num_iter=2,
                dataset='test',
                setname='icvl',
                dataset_path_prefix=constants.Data_Path,
                source_name='_iter1_whlimg_holi_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                batch_size = 133,
                jnt_idx = [9],
                c1=24,
                c2=48,
                patch_size=40,
                h1_out_factor=1,
                h2_out_factor=1,
                model_save_path = 'param_cost_egoff_iter2_bw9_r012_24_48_1_1_adam_lm99',
                offset_save_path ='_egoff_iter2_bw9_r012_24_48_1_1_adam_lm99' )