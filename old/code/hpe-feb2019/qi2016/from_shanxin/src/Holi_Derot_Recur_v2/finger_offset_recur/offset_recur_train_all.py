__author__ = 'QiYE'
import theano
import theano.tensor as T
import numpy

from src.utils import constants
from src.Holi_Derot_Recur_v2.finger_offset_recur.load_data import  load_data_multi_offset
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import adam_update,set_params


def train_model(all_idx,setname,dataset_path_prefix,source_name,batch_size,patch_size,c1,c2,h1_out_factor,h2_out_factor,lamda):
    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')# x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')

    img_size_0 = 48
    img_size_1 = 24
    img_size_2 = 14
    model_info='offset_r012_21jnts_derot_patch%d'%(patch_size)
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
    beta1=0
    beta2=0
    cost = model.cost(Y)
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


    ini_param_value =[]
    for param_i in model.params:
        ini_param_value.append(param_i.get_value())

    for jnt_idx in all_idx:
        model_info='offset_jnt%d_r012_patch%d'%(jnt_idx,patch_size)
        print model_info, constants.OUT_DIM

        dataset = 'train'
        src_path = '%sdata/%s/holi_derot_recur_v2/bw_offset/best/'%(constants.Data_Path,setname)
        path = '%s%s%s.h5'%(src_path,dataset,source_name)
        print 'source path',path

        train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi_offset(batch_size,path,jnt_idx=jnt_idx,is_shuffle=True,
                                                                                                patch_size=patch_size,patch_pad_width=4,hand_width=96,hand_pad_width=0)
        n_train_batches = train_set_x0.shape[0]/ batch_size

        print 'n_train_batches', n_train_batches

        dataset = 'test'
        path = '%s%s%s.h5'%(src_path,dataset,source_name)
        test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_offset(batch_size,path,jnt_idx=jnt_idx,is_shuffle=True,
                                                                                            patch_size=patch_size,patch_pad_width=4,hand_width=96,hand_pad_width=0)
        n_test_batches = test_set_x0.shape[0]/ batch_size

        print 'n_test_batches', n_test_batches

        for param_i, ini_params_v in zip(model.params, ini_param_value):
            param_i.set_value(ini_params_v)
        print 'intial parameters value set'



        n_epochs =80
        epoch = 0
        test_cost=[0]
        train_cost=[0]
        done_looping=False
        save_path = '%sdata/%s/holi_derot_recur_v2/fingers/best/jnt%s_'%(dataset_path_prefix,setname,jnt_idx)
        drop = numpy.cast['int32'](0)
        print 'dropout', drop

        model.save_adam(path=save_path,c00=c1,c01=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,lamda=learning_rate.get_value()*1000000,epoch=epoch,
                       train_cost=train_cost,test_cost=test_cost)
        cost_tmp=999999
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
                print '                                              save min cost',test_cost[-1]
                cost_tmp=test_cost[-1]
                model.save_adam(path=save_path,c00=c1,c01=c2,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,lamda=learning_rate.get_value()*1000000,epoch=epoch,
                       train_cost=train_cost,test_cost=test_cost)
            print 'test ', test_cost[-1],'  train', train_cost[-1]
            cost_nbatch = 0
            # t0 = time.clock()
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

    train_model(all_idx = [2,3,4],
        setname='icvl',
                dataset_path_prefix=constants.Data_Path,
                source_name='_iter1_whlimg_holi_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                lamda = 0.001,
                batch_size = 100,
                patch_size=40,
                c1=8,
                c2=16,
                h1_out_factor=1,
                h2_out_factor=1)
    # train_model(all_idx = [6,7,8],
    #     setname='icvl',
    #             dataset_path_prefix=constants.Data_Path,
    #             source_name='_iter1_whlimg_holi_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             lamda = 0.001,
    #             batch_size = 100,
    #             patch_size=40,
    #             c1=8,
    #             c2=16,
    #             h1_out_factor=1,
    #             h2_out_factor=1)
    # train_model(all_idx = [10,11,12],
    #     setname='icvl',
    #             dataset_path_prefix=constants.Data_Path,
    #             source_name='_iter1_whlimg_holi_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             lamda = 0.001,
    #             batch_size = 100,
    #             patch_size=40,
    #             c1=8,
    #             c2=16,
    #             h1_out_factor=1,
    #             h2_out_factor=1)
    # train_model(all_idx = [14,15,16,18,19,20],
    #     setname='icvl',
    #             dataset_path_prefix=constants.Data_Path,
    #             source_name='_iter1_whlimg_holi_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             lamda = 0.001,
    #             batch_size = 100,
    #             patch_size=40,
    #             c1=8,
    #             c2=16,
    #             h1_out_factor=1,
    #             h2_out_factor=1)
    #
    #
    # jnt_idx = [0,1,5,9 ,13,17]

    # idx=[19]
    # idx=[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]
    #
    # train_model(all_idx = idx,
    #     setname='nyu',
    #             dataset_path_prefix=constants.Data_Path,
    #             source_name='_iter1_whlimg_holi_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             lamda = 0.002,
    #             batch_size = 100,
    #             patch_size=40,
    #             c1=14,
    #             c2=28,
    #             h1_out_factor=2,
    #             h2_out_factor=4)

    # idx=[3]
    # # idx=[7,6]
    # # idx=[11,10]
    # # idx=[15,14]
    # # idx=[19,18]
    # train_model(all_idx = idx,
    #     setname='msrc',
    #             dataset_path_prefix=constants.Data_Path,
    #             source_name='_iter1_whlimg_holi_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    #             lamda = 0.004,
    #             batch_size = 100,
    #             patch_size=40,
    #             c1=14,
    #             c2=28,
    #             h1_out_factor=2,
    #             h2_out_factor=4)