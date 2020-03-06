from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi_top_uvd_normalized
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import adam_update,set_params


def train_model(setname, dataset_path_prefix,source_name,prev_jnt_name,batch_size,jnt_idx,patch_size,offset_depth_range,c1,c2,h1_out_factor,h2_out_factor,lamda):
    beta1 = 0.6
    beta2 =0.6
    print 'offset_depth_range ',offset_depth_range
    model_info='offset_top%d_patch%d_beta%d'%(jnt_idx[0],patch_size,beta1*10)
    print model_info, constants.OUT_DIM

    dataset = 'train'
    src_path = '%sdata/%s/hier_derot_recur_v2/bw_offset/best/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    print 'source name',path
    direct = '%sdata/%s/hier_derot_recur_v2/final_xyz_uvd/'%(dataset_path_prefix,setname)
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    print 'prev_jnt_path',prev_jnt_path
    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi_top_uvd_normalized(batch_size,path,prev_jnt_path,jnt_idx=jnt_idx,is_shuffle=True,
                                                                                            patch_size=patch_size, patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'num_sample, n_train_batches',train_set_x0.shape[0], n_train_batches

    dataset = 'test'
    path = '%s%s%s.h5'%(src_path,dataset,source_name)

    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_top_uvd_normalized(batch_size,path,prev_jnt_path,jnt_idx=jnt_idx,is_shuffle=True,
                                                                                        patch_size=patch_size, patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    n_test_batches = test_set_x0.shape[0]/ batch_size
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

    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )
    m=[]
    v=[]
    for param in  model.params:
        m.append(theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.)))
        v.append(theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.)))

    assert lamda >= 0. and lamda < 1.
    beta1_t=theano.shared(numpy.cast[theano.config.floatX](beta1*beta1), name='momentum')
    beta2_t=theano.shared(numpy.cast[theano.config.floatX](beta2*beta2), name='momentum')

    updates = adam_update(model,cost,m=m,v=v,beta1=beta1,beta2=beta2,beta1_t=beta1_t,beta2_t=beta2_t,learning_rate=learning_rate)

    print 'beta1,beta2,learning_rate', beta1, beta2, lamda

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,on_unused_input='ignore')

    n_epochs =200
    epoch = 0
    test_cost=[0]
    train_cost=[0]
    done_looping=False
    cost_tmp=9999
    save_path = '%sdata/%s/hier_derot_recur_v2/top/'%(dataset_path_prefix,setname)
    # model.save(path=save_path,c00=c1,c01=c2,c10=c1,c11=c2,c20=c1,c21=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,gamma=gamma*100000,lamda=lamda*100000,yita=yita*100000,epoch=epoch,train_cost=train_cost,test_cost=test_cost)
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    while (epoch < n_epochs) and (not done_looping):

        epoch +=1
        print 'traing @ epoch = ', epoch
        cost_nbatch = 0
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_nbatch+=cost_ij

        test_cost.append(cost_nbatch/n_test_batches)

        if test_cost[-1]<cost_tmp:
            print '                                        save min cost',test_cost[-1]
            cost_tmp=test_cost[-1]
            model.save_adam(path=save_path,c00=c1,c01=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,lamda=learning_rate.get_value()*1000000,epoch=epoch,
                   train_cost=train_cost,test_cost=test_cost)
        print 'test ', test_cost[-1],'  train', train_cost[-1]
        cost_nbatch = 0

        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = train_model(x0,x1,x2,drop, y)
            cost_nbatch+=cost_ij
        train_cost.append(cost_nbatch/n_train_batches)

if __name__ == '__main__':



    idx =3
    train_model(setname='icvl',
                dataset_path_prefix=constants.Data_Path,
                source_name='_iter1_whlimg_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                prev_jnt_name='_absuvd0_offset_mid%d_patch40_beta6_24_48_1_1_adam_lm3'%(idx-1),
                lamda = 0.000003,
                batch_size = 100,
                jnt_idx = [idx],
                patch_size=40,
                offset_depth_range=0.4,
                c1=24,
                c2=48,
                h1_out_factor=1,
                h2_out_factor=1)
    #
    # idx_all = [2,6,10,14,18]
    # for i ,idx in enumerate(idx_all):
    #     test_model(dataset='train',setname='icvl',
    #                dataset_path_prefix=constants.Data_Path,
    #                 source_name='_iter1_whlimg_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                 batch_size = 8,
    #                 jnt_idx = [idx],
    #                 patch_size=40,
    #                 offset_depth_range=0.4,
    #                 c1=24,
    #                 c2=48,
    #                 h1_out_factor=1,
    #                 h2_out_factor=1,
    #                          model_save_path = '_offset_mid%d_patch40_beta6_24_48_1_1_adam_lm3'%idx,
    #                          offset_save_path = '_offset_mid%d_patch40_beta6_24_48_1_1_adam_lm3' %idx)
    # jnt_idx = [3,7,11,15,19]