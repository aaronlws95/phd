__author__ = 'QiYE'

import numpy
from sklearn.utils import shuffle
import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from utils import data_augmentation


def fit_palm_stage_0(batch_size,n_epochs,model,
          train_img0,train_img1,train_img2,train_target,
          test_img0,test_img1,test_img2,test_target,model_filepath,history_filepath):
    n_train_batches=int(train_img0.shape[0]/batch_size)
    train_idx=range(train_img0.shape[0])
    epoch = 0

    done_looping=False

    best_lost=999
    test_cost=[best_lost]
    train_cost=[best_lost]
    # train_idx=range(train_idx.shape[0])
    validfreq = 400
    # validfreq = 20
    num_iter=0
    while (epoch < n_epochs) and (not done_looping):
        epoch +=1
        print('traing @ epoch = ', epoch)
        train_idx=shuffle(train_idx)
        for minibatch_index in range(n_train_batches):
            num_iter+=1
            batch_idx = train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x0,x1,x2,y=data_augmentation.augment_data_3d_mega_rot_scale(train_img0[batch_idx],train_img1[batch_idx],train_img2[batch_idx], train_target[batch_idx])
            # x0=train_img0[batch_idx]
            # x1=train_img1[batch_idx]
            # x2=train_img2[batch_idx]
            # y=train_target[batch_idx]
            out = model.train_on_batch(x={'input0':x0,'input1':x1,'input2':x2},y=y)

            # print(version,'iter epoch,minibatch',num_iter, epoch,minibatch_index)
            # print('train',out,'cur',test_cost[-1],'best', best_lost)

            if numpy.isnan(out).any():
                exit('nan,%d,%d'%(epoch,minibatch_index))

            if (num_iter+1)%int(validfreq)==0:
                model.save_weights('%s_epoch'%model_filepath, overwrite=True)
                numpy.save('%s_epoch'%history_filepath,[train_cost,test_cost])
                val_loss = model.evaluate(x={'input0':test_img0,'input1':test_img1,'input2':test_img2},y=test_target,batch_size=batch_size)

                if numpy.isinf(val_loss).any():
                    model.save_weights('%s_inf'%model_filepath, overwrite=True)
                    numpy.save('%s_inf'%history_filepath,[train_cost,test_cost])
                print('\n')
                print('epoch',epoch, 'minibatch_index',minibatch_index, 'train_loss',out,'val_loss',val_loss)
                test_cost.append(val_loss)
                train_cost.append(out)
                if val_loss<best_lost:
                    print('-'*30,model_filepath,'best val_loss',val_loss)
                    best_lost=val_loss
                    model.save_weights(model_filepath, overwrite=True)
                    numpy.save(history_filepath,[train_cost,test_cost])




def fit_palm_stage_1(batch_size,n_epochs,model,
          train_img0,train_target,train_pred_palm,
          test_img0,test_target,test_pred_palm,model_filepath,history_filepath,palm_jnt):

    test_x0,test_x1,test_y=data_augmentation.get_hand_part_for_palm(r0=test_img0,
                                                        gr_uvd=test_target,pred_uvd=test_pred_palm,jnt=palm_jnt,if_aug=False)

    n_train_batches=int(train_img0.shape[0]/batch_size)
    train_idx=range(train_img0.shape[0])
    epoch = 0

    done_looping=False

    best_lost=999
    test_cost=[best_lost]
    train_cost=[best_lost]
    # train_idx=range(train_idx.shape[0])
    # validfreq = 400
    validfreq = 100
    num_iter=0
    while (epoch < n_epochs) and (not done_looping):
        epoch +=1
        print('traing @ epoch = ', epoch)
        train_idx=shuffle(train_idx)
        for minibatch_index in range(n_train_batches):
            num_iter+=1
            batch_idx = train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x0,x1,y=data_augmentation.get_hand_part_for_palm(r0=train_img0[batch_idx],
                                                                gr_uvd=train_target[batch_idx],pred_uvd=train_pred_palm[batch_idx],
                                                                jnt=palm_jnt,if_aug=True)
            out = model.train_on_batch(x={'input0':x0,'input1':x1},y=y)

            if numpy.isnan(out).any():
                exit('nan,%d,%d'%(epoch,minibatch_index))

            if (num_iter+1)%int(validfreq)==0:
                model.save_weights('%s_epoch'%model_filepath, overwrite=True)
                numpy.save('%s_epoch'%history_filepath,[train_cost,test_cost])
                val_loss = model.evaluate(x={'input0':test_x0,'input1':test_x1},y=test_y,batch_size=batch_size)

                if numpy.isinf(val_loss).any():
                    model.save_weights('%s_inf'%model_filepath, overwrite=True)
                    numpy.save('%s_inf'%history_filepath,[train_cost,test_cost])
                print('\n')
                print('epoch',epoch, 'minibatch_index',minibatch_index, 'train_loss',out,'val_loss',val_loss)
                test_cost.append(val_loss)
                train_cost.append(out)
                if val_loss<best_lost:
                    print('-'*30,model_filepath,'best val_loss',val_loss)
                    best_lost=val_loss
                    model.save_weights(model_filepath, overwrite=True)
                    numpy.save(history_filepath,[train_cost,test_cost])


def fit_palm_stage_1_holi(batch_size,n_epochs,model,
          train_img0,train_target,train_pred_palm,train_uvd_centre,
          test_img0,test_target,test_pred_palm,test_uvd_centre,model_filepath,history_filepath):

    test_x0,test_x1,test_x2,test_y=data_augmentation.get_img_for_palm_holi(r0=test_img0,
                                                        gr_uvd=test_target,pred_uvd=test_pred_palm,uvd_hand_center=test_uvd_centre,if_aug=False)

    n_train_batches=int(train_img0.shape[0]/batch_size)
    train_idx=range(train_img0.shape[0])
    epoch = 0

    done_looping=False

    best_lost=999
    test_cost=[best_lost]
    train_cost=[best_lost]
    # train_idx=range(train_idx.shape[0])
    validfreq = 400
    # validfreq = 100
    num_iter=0
    while (epoch < n_epochs) and (not done_looping):
        epoch +=1
        print('traing @ epoch = ', epoch)
        train_idx=shuffle(train_idx)
        for minibatch_index in range(n_train_batches):
            num_iter+=1
            batch_idx = train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x0,x1,x2,y=data_augmentation.get_img_for_palm_holi(r0=train_img0[batch_idx],
                                                                gr_uvd=train_target[batch_idx],pred_uvd=train_pred_palm[batch_idx],
                                                                uvd_hand_center=train_uvd_centre[batch_idx],if_aug=True)
            out = model.train_on_batch(x={'input0':x0,'input1':x1,'input2':x2},y=y)


            if numpy.isnan(out).any():
                exit('nan,%d,%d'%(epoch,minibatch_index))

            if (num_iter+1)%int(validfreq)==0:
                model.save_weights('%s_epoch'%model_filepath, overwrite=True)
                numpy.save('%s_epoch'%history_filepath,[train_cost,test_cost])
                val_loss = model.evaluate(x={'input0':test_x0,'input1':test_x1,'input2':test_x2},y=test_y,batch_size=batch_size)

                if numpy.isinf(val_loss).any():
                    model.save_weights('%s_inf'%model_filepath, overwrite=True)
                    numpy.save('%s_inf'%history_filepath,[train_cost,test_cost])
                print('\n')
                print('epoch',epoch, 'minibatch_index',minibatch_index, 'train_loss',out,'val_loss',val_loss)
                test_cost.append(val_loss)
                train_cost.append(out)
                if val_loss<best_lost:
                    print('-'*30,model_filepath,'best val_loss',val_loss)
                    best_lost=val_loss
                    model.save_weights(model_filepath, overwrite=True)
                    numpy.save(history_filepath,[train_cost,test_cost])



def fit_pip_stage_0(batch_size,n_epochs,model,
          train_img0,train_target,train_pred_palm,train_jnt_uvd_in_prev_layer,
          test_img0,test_target,test_pred_palm,test_jnt_uvd_in_prev_layer,model_filepath,history_filepath,if_aug,aug_trans,aug_rot):

    test_x0,test_x1,test_y=data_augmentation.get_crop_for_finger_part_s0(r0=test_img0,
                                                        gr_uvd=test_target,pred_uvd=test_pred_palm,
                                                        jnt_uvd_in_prev_layer=test_jnt_uvd_in_prev_layer,if_aug=False)

    n_train_batches=int(train_img0.shape[0]/batch_size)
    train_idx=range(train_img0.shape[0])
    epoch = 0

    done_looping=False

    best_lost=999
    test_cost=[best_lost]
    train_cost=[best_lost]
    # train_idx=range(train_idx.shape[0])
    validfreq = 200
    # validfreq = 20
    num_iter=0
    while (epoch < n_epochs) and (not done_looping):
        epoch +=1
        print('traing @ epoch = ', epoch)
        train_idx=shuffle(train_idx)
        for minibatch_index in range(n_train_batches):
            num_iter+=1
            batch_idx = train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x0,x1,y=data_augmentation.get_crop_for_finger_part_s0(r0=train_img0[batch_idx], gr_uvd=train_target[batch_idx], pred_uvd=train_pred_palm[batch_idx], jnt_uvd_in_prev_layer=train_jnt_uvd_in_prev_layer[batch_idx], if_aug=if_aug,aug_rot=aug_rot,aug_trans=aug_trans)

            out = model.train_on_batch(x={'input0':x0,'input1':x1},y=y)


            if numpy.isnan(out).any():
                exit('nan,%d,%d'%(epoch,minibatch_index))

            if (num_iter+1)%int(validfreq)==0:
                model.save_weights('%s_epoch'%model_filepath, overwrite=True)
                numpy.save('%s_epoch'%history_filepath,[train_cost,test_cost])
                val_loss = model.evaluate(x={'input0':test_x0,'input1':test_x1},y=test_y,batch_size=batch_size)

                if numpy.isinf(val_loss).any():
                    model.save_weights('%s_inf'%model_filepath, overwrite=True)
                    numpy.save('%s_inf'%history_filepath,[train_cost,test_cost])
                print('\n')
                print('epoch',epoch, 'minibatch_index',minibatch_index, 'train_loss',out,'val_loss',val_loss)
                test_cost.append(val_loss)
                train_cost.append(out)
                if val_loss<best_lost:
                    print('-'*30,model_filepath,'best val_loss',val_loss)
                    best_lost=val_loss
                    model.save_weights(model_filepath, overwrite=True)
                    numpy.save(history_filepath,[train_cost,test_cost])




def fit_dtip_stage_0(batch_size,n_epochs,model,
          train_img0,train_target,train_pred_palm,train_jnt_uvd_in_prev_layer,
          test_img0,test_target,test_pred_palm,test_jnt_uvd_in_prev_layer,model_filepath,history_filepath,
          if_aug=True,aug_rot=15,aug_trans=0.05):

    test_x0,test_x1,test_y=data_augmentation.get_crop_for_finger_part_s0(r0=test_img0,
                                                        gr_uvd=test_target,pred_uvd=test_pred_palm,
                                                        jnt_uvd_in_prev_layer=test_jnt_uvd_in_prev_layer,if_aug=False)

    n_train_batches=int(train_img0.shape[0]/batch_size)
    train_idx=range(train_img0.shape[0])
    epoch = 0

    done_looping=False

    best_lost=999
    test_cost=[best_lost]
    train_cost=[best_lost]
    # train_idx=range(train_idx.shape[0])
    validfreq = 200
    # validfreq = 20
    num_iter=0
    while (epoch < n_epochs) and (not done_looping):
        epoch +=1
        print('traing @ epoch = ', epoch)
        train_idx=shuffle(train_idx)
        for minibatch_index in range(n_train_batches):
            num_iter+=1
            batch_idx = train_idx[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            x0,x1,y=data_augmentation.get_crop_for_finger_part_s0(r0=train_img0[batch_idx],
                                                                gr_uvd=train_target[batch_idx],
                                                                pred_uvd=train_pred_palm[batch_idx],
                                                                jnt_uvd_in_prev_layer=train_jnt_uvd_in_prev_layer[batch_idx],
                                                                if_aug=if_aug,aug_rot=aug_rot,aug_trans=aug_trans)
            out = model.train_on_batch(x={'input0':x0,'input1':x1},y=y)


            if numpy.isnan(out).any():
                exit('nan,%d,%d'%(epoch,minibatch_index))

            if (num_iter+1)%int(validfreq)==0:
                model.save_weights('%s_epoch'%model_filepath, overwrite=True)
                numpy.save('%s_epoch'%history_filepath,[train_cost,test_cost])
                val_loss = model.evaluate(x={'input0':test_x0,'input1':test_x1},y=test_y,batch_size=batch_size)

                if numpy.isinf(val_loss).any():
                    model.save_weights('%s_inf'%model_filepath, overwrite=True)
                    numpy.save('%s_inf'%history_filepath,[train_cost,test_cost])
                print('\n')
                print('epoch',epoch, 'minibatch_index',minibatch_index, 'train_loss',out,'val_loss',val_loss)
                test_cost.append(val_loss)
                train_cost.append(out)
                if val_loss<best_lost:
                    print('-'*30,model_filepath,'best val_loss',val_loss)
                    best_lost=val_loss
                    model.save_weights(model_filepath, overwrite=True)
                    numpy.save(history_filepath,[train_cost,test_cost])
