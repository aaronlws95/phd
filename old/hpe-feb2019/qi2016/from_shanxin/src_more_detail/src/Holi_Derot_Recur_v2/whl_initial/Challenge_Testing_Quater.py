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

def train_model_conv3(setname, cross_test_set,num_test_sample,
                      dataset_path_prefix,source_name,batch_size,jnt_idx,c1,c2,c3,h1_out_factor,h2_out_factor,lamda,model_name,model_size,bbsize,
                      testingdata_name):
    print 'batch_size', batch_size
    model_info='uvd_whl_r012_21jnts'
    print model_info
    # src_path = '%s/%s/'%(dataset_path_prefix,setname)
    trainortest = 'test'
    # path = '%s%s_%s%s.h5'%(src_path,cross_test_set,trainortest,source_name)
    path = testingdata_name

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

    set_adam_params('%s/%s/param_cost_%s.npy'%(dataset_path_prefix,setname,model_name),model.params)
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
    result_save_path = "%s/%s/result_%s"%(dataset_path_prefix,setname,model_size)
    numpy.save('%s_%s_%s_bb%d_pred_norm_uvd_%s'%(result_save_path,cross_test_set,trainortest,bbsize,model_name),pred_y[0:num_test_sample])

    xyz,xyz_gt,err = error.normuvd2xyz_mega(cross_test_set, path, pred_y[0:num_test_sample])
    print 'mean err ', numpy.mean(numpy.mean(err))
    print 'mean err for 21 joints', numpy.mean(err,axis=0)

    scipy.io.savemat('%s_%s_%s_bb%d_xyz_%s'%(result_save_path,cross_test_set,trainortest,bbsize,model_name),
                     {'xyz_pred':xyz,'xyz_jnt_gt':xyz_gt})




if __name__ == '__main__':

    bbsize=260
    print 'bbsize=',bbsize

    # # for validation
    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge.h5'
    # setname = 'challenge'
    # cross_test_set = 'test'
    # num_test_sample = 87155

    # Guillermo
    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Seen_Guillermo.h5'
    # setname = 'challenge'
    # cross_test_set = 'Ego_Seen_Guillermo'
    # num_test_sample = 33904

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Seen_Guillermo.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Seen_Guillermo'
    # num_test_sample = 49810

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Seen_Pamela.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Seen_Pamela'
    # num_test_sample = 49695

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Seen_seung.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Seen_seung'
    # num_test_sample = 48962

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Seen_seung.h5'
    # setname = 'challenge'
    # cross_test_set = 'Ego_Seen_seung'
    # num_test_sample = 33791

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Seen_ShanxinV7.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Seen_ShanxinV7'
    # num_test_sample = 48273

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Seen_ShanxinV7.h5'
    # setname = 'challenge'
    # cross_test_set = 'Ego_Seen_ShanxinV7'
    # num_test_sample = 33755

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Seen_Xinghao.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Seen_Xinghao'
    # num_test_sample = 51830
    #
    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Seen_Xinghao.h5'
    # setname = 'challenge'
    # cross_test_set = 'Ego_Seen_Xinghao'
    # num_test_sample = 33754


    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Unseen_Caner.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Unseen_Caner'
    # num_test_sample = 34535

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Unseen_Caner.h5'
    # setname = 'challenge'
    # cross_test_set = 'Ego_Unseen_Caner'
    # num_test_sample = 33224

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Unseen_Patrick.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Unseen_Patrick'
    # num_test_sample = 48727

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Unseen_Patrick.h5'
    # setname = 'challenge'
    # cross_test_set = 'Ego_Unseen_Patrick'
    # num_test_sample = 33878

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Unseen_Qi.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Unseen_Qi'
    # num_test_sample = 46214

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Unseen_Qi.h5'
    # setname = 'challenge'
    # cross_test_set = 'Ego_Unseen_Qi'
    # num_test_sample = 33456

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Unseen_sara.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Unseen_sara'
    # num_test_sample = 48029

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Unseen_sara.h5'
    # setname = 'challenge'
    # cross_test_set = 'Ego_Unseen_sara'
    # num_test_sample = 31773

    # testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Free_Unseen_vassileios.h5'
    # setname = 'challenge'
    # cross_test_set = 'Free_Unseen_vassileios'
    # num_test_sample = 42178

    testingdata_name = '/media/Data/shanxin/StructuralHand/Mega/data/challenge/Challenge_norm_hand_uvd_rootmid_testingloc_Challenge_Ego_Unseen_vassileios.h5'
    setname = 'challenge'
    cross_test_set = 'Ego_Unseen_vassileios'
    num_test_sample = 33928



    train_model_conv3(
                    setname=setname,
                    cross_test_set = cross_test_set,
                    num_test_sample = num_test_sample,
                    dataset_path_prefix=constants.Data_Path,
                    model_name='uvd_Quater_whl_r012_21jnts_64_96_128_1_2_adam_lm99',
                    model_size='full',
                    bbsize=bbsize,
                    source_name='_norm_hand_uvd_rootmid',
                    lamda=0.0001,
                    batch_size = 64,
                    jnt_idx = range(0,21,1),
                    c1=64,
                    c2=96,
                    c3=128,
                    h1_out_factor=1,
                    h2_out_factor=2,
                    testingdata_name = testingdata_name)

