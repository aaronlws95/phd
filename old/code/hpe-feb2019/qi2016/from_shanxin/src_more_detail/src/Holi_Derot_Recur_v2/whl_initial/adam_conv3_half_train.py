__author__ = 'QiYE'
import time
import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi
from src.Model.CNN_Model import CNN_Model_multi3_conv3
from src.Model.Train import adam_update,set_adam_params
from src.utils import constants
def train_model_conv3(setname, dataset_path_prefix,source_name,batch_size,jnt_idx,c1,c2,c3,h1_out_factor,h2_out_factor,lamda):
    print 'batch_size', batch_size
    model_info='uvd_whl_r012_21jnts_half'
    print model_info
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    dataset = 'train'
    path = '%smega_%s%s.h5'%(src_path,dataset,source_name)


    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi(batch_size=batch_size,path=path,is_shuffle=True,jnt_idx=jnt_idx)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    img_size_0 = train_set_x0.shape[2]
    img_size_1 = train_set_x1.shape[2]
    img_size_2 = train_set_x2.shape[2]
    print 'n_train_batches', n_train_batches

    dataset = 'test'
    path = '%smega_%s%s.h5'%(src_path,dataset,source_name)

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi(batch_size=batch_size,path=path,is_shuffle=True, jnt_idx=jnt_idx)
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


    beta1 = 0.9
    beta2 =0.999
    # Convert the learning rate into a shared variable to adapte the learning rate during training.
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](lamda) )
    print 'beta1,beta2,lamda',beta1,beta2,lamda
    interupt = 1
    if interupt == 0:

        m=[]
        v=[]
        for param in  model.params:
            m.append(theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.)))
            v.append(theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.)))

        assert lamda >= 0. and lamda < 1.
        beta1_t=theano.shared(numpy.cast[theano.config.floatX](beta1), name='momentum')
        beta2_t=theano.shared(numpy.cast[theano.config.floatX](beta2), name='momentum')
        print 'beta1_t', beta1_t.get_value(),'beta2_t', beta2_t.get_value()
    else:
        m,v,beta1_t,beta2_t = set_adam_params('%sdata/%s/whole/param_cost_uvd_whl_r012_21jnts_64_96_128_1_2_adam_lm99_bf.npy'%
                                              (dataset_path_prefix,setname),model.params)



    updates = adam_update(model,cost,
                          m=m,v=v,beta1=beta1,beta2=beta2,beta1_t=beta1_t,beta2_t=beta2_t,
                          learning_rate=learning_rate)

    train_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,updates=updates,on_unused_input='ignore')
    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=cost,on_unused_input='ignore')


    n_epochs =10000
    epoch = 0
    cost_tmp=999
    test_cost=[0]
    train_cost=[0]

    done_looping=False
    save_path =  '%sdata/%s/whole/'%(dataset_path_prefix,setname)
    drop = numpy.cast['int32'](0)
    print 'dropout', drop
    #
    model.save_adam(path=save_path,
                    m=m,v=v,beta1_t=beta1_t,beta2_t=beta2_t,
                    c00=c1,c01=c2,c02=c3,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,
                    lamda=learning_rate.get_value()*1000000,epoch=epoch,
                   train_cost=train_cost,test_cost=test_cost)

    while (epoch < n_epochs) and (not done_looping):

        cost_nbatch = 0
        start = time.clock()
        for minibatch_index in xrange(n_test_batches):
            # print minibatch_index
            x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
            cost_nbatch+=cost_ij
        print '                                              test time per epoch in minutes', (time.clock()-start)/60.0
        test_cost.append(cost_nbatch/n_test_batches)
        if test_cost[-1]<cost_tmp:
            print '                                                           save min cost',test_cost[-1], ' ',save_path
            cost_tmp=test_cost[-1]
            model.save_adam(path=save_path,
                            m=m,v=v,beta1_t=beta1_t,beta2_t=beta2_t,
                            c00=c1,c01=c2,c02=c3,h1_out_factor=h1_out_factor,h2_out_factor=h2_out_factor,
                            lamda=learning_rate.get_value()*1000000,epoch=epoch,
                            train_cost=train_cost,test_cost=test_cost)

        cost_nbatch = 0
        start = time.clock()
        for minibatch_index in xrange(n_train_batches):
            # print minibatch_index
            x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            cost_ij = train_model(x0,x1,x2,drop, y)
            cost_nbatch+=cost_ij
        print '                                                train time per epoch in minutes', (time.clock()-start)/60.0
        train_cost.append(cost_nbatch/n_train_batches)


        beta1_t.set_value(numpy.cast[theano.config.floatX](beta1_t.get_value()*beta1))
        beta2_t.set_value(numpy.cast[theano.config.floatX](beta2_t.get_value()*beta2))

        epoch +=1
        print 'traing @ epoch = ', epoch
        print 'train cost:',train_cost[-1], '    test cost:',test_cost[-1]

if __name__ == '__main__':


    train_model_conv3(setname='mega',
                     dataset_path_prefix=constants.Data_Path,
            source_name='_norm_hand_uvd_rootmid_half',
                lamda=0.0003,
                batch_size = 128,
                jnt_idx = range(0,21,1),
                c1=64,
                c2=96,
                c3=128,
                h1_out_factor=1,
                h2_out_factor=2)

    # train_model_conv3(setname='nyu',
    #             source_name='_nyu_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             lamda = 0.003,
    #             batch_size = 100,
    #             dataset_path_prefix=constants.Data_Path,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             c1=16,
    #             c2=32,
    #             c3=64,
    #             h1_out_factor=2,
    #             h2_out_factor=4)
    #



