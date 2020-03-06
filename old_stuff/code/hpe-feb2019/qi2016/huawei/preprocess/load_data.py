from __future__ import print_function
from sklearn.utils import shuffle
from PIL import Image
import numpy
import matplotlib.pyplot as plt
from ..utils import xyz_uvd
from skimage.transform import resize


def get_filenames_labels(dataset_dir):

    xyz_jnt_gt=[]
    file_name = []
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
    with open('%s/Training_Annotation.txt'%(dataset_dir), mode='r',encoding='utf-8',newline='') as f:
        for line in f:
            part = line.split('\t')
            file_name.append(part[0])
            xyz_jnt_gt.append(part[1:64])
    f.close()
    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float64')

    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt =xyz_uvd.xyz2uvd(xyz=xyz_jnt_gt,setname='mega')

    return uvd_jnt_gt,xyz_jnt_gt,numpy.array(file_name)

def generate_fullimg_mask_from_file_unet(path,img_file_name,uvd,batch_size):
    centerU=315.944855
    phyMargin=50.0
    padWidth=200
    img_rows=480
    img_cols=640

    num_imgs=len(img_file_name)
    idx = numpy.arange(len(img_file_name))
    num = (batch_size - num_imgs%batch_size)%batch_size
    idx = numpy.concatenate([idx,idx[0:num]],axis=0)
    n_batches = int(idx.shape[0]/batch_size)

    x0 = numpy.zeros((batch_size,img_rows,img_cols,1),dtype='float32')
    y = numpy.zeros((batch_size,img_rows,img_cols,1),dtype='uint8')

    while 1:
        idx= shuffle(idx,random_state=0)
        for minibatch_index in range(n_batches):
            # print('minibatch_index',minibatch_index)
            slice_idx = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
            for mi, cur_idx in enumerate(list(idx[slice_idx])):
                cur_file_name = img_file_name[cur_idx]
                cur_uvd = uvd[cur_idx]

                bb = numpy.array([(phyMargin,phyMargin,cur_uvd[9,2])])
                bbox_uvd = xyz_uvd.xyz2uvd(setname='mega',xyz=bb)
                margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))

                roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
                depth = numpy.asarray(roiDepth, dtype='uint16')
                x0[mi,:,:,0]=depth/2000.0

                axis_bounds = numpy.array([numpy.min(cur_uvd[:, 0]), numpy.max(cur_uvd[:, 0]),
                                           numpy.min(cur_uvd[:, 1]), numpy.max(cur_uvd[:, 1]),
                                           numpy.min(cur_uvd[:, 2]), numpy.max(cur_uvd[:, 2])],dtype='int32')

                tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
                tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth

                crop = tmpDepth[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                       axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth]
                loc = numpy.where(numpy.logical_and(crop>axis_bounds[4]-phyMargin,crop<axis_bounds[5]+phyMargin))
                cropmask=numpy.zeros_like(crop)

                cropmask[loc]=1
                orimask = numpy.zeros_like(tmpDepth,dtype='uint8')
                orimask[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                               axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth] =cropmask
                y[mi,:,:,0] = orimask[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]

            yield (x0,y)


def generate_train(path,img_file_name,uvd,batch_size):
    num_imgs=uvd.shape[0]
    idx = numpy.arange(num_imgs)
    print('train.num',num_imgs)
    n_batches=int(idx.shape[0]/batch_size)


    phyMargin=50.0
    padWidth=200
    centerU=315.944855
    setname='mega'

    while True:
        idx= shuffle(idx,random_state=0)
        for minibatch_index in range(n_batches):
            # print('minibatch_index',minibatch_index)
            slice_idx = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
            all_input = numpy.zeros((batch_size,128,160,1),dtype='float32')
            all_mask=numpy.zeros((batch_size,128,160,1),dtype='uint8')
            for mi, cur_idx in enumerate(list(idx[slice_idx])):
                cur_file_name = img_file_name[cur_idx]
                cur_uvd = uvd[cur_idx]
                bb = numpy.array([(phyMargin,phyMargin,cur_uvd[9,2])])
                bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
                margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))
                roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
                depth = numpy.asarray(roiDepth, dtype='uint16')

                axis_bounds = numpy.array([numpy.min(cur_uvd[:, 0]), numpy.max(cur_uvd[:, 0]),
                                           numpy.min(cur_uvd[:, 1]), numpy.max(cur_uvd[:, 1]),
                                           numpy.min(cur_uvd[:, 2]), numpy.max(cur_uvd[:, 2])],dtype='int32')

                tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
                tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth

                crop = tmpDepth[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                       axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth]
                loc = numpy.where(numpy.logical_and(crop>axis_bounds[4]-phyMargin,crop<axis_bounds[5]+phyMargin))
                cropmask=numpy.zeros_like(crop)

                cropmask[loc]=1
                orimask = numpy.zeros_like(tmpDepth,dtype='uint8')
                orimask[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                               axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth] =cropmask
                orimask = orimask[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]
                orimask = resize(orimask,(120,160), order=3,preserve_range=True)
                orimask[numpy.where(orimask>0)]=1
                all_mask[mi,4:124,:,0]=orimask
                all_input[mi,4:124,:,0]=resize(depth,(120,160), order=3,preserve_range=True)/2000.0
            yield (all_input,all_mask)


def generate_downsample_img_mask_from_file_unet_aug(path,img_file_name,uvd,batch_size):
    num_imgs=uvd.shape[0]
    idx = numpy.arange(num_imgs)
    print('train.num',num_imgs)
    n_batches=int(idx.shape[0]/batch_size)


    phyMargin=50.0
    padWidth=200
    centerU=315.944855
    setname='mega'

    while True:
        idx= shuffle(idx,random_state=0)
        for minibatch_index in range(n_batches):
            # print('minibatch_index',minibatch_index)
            slice_idx = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
            all_input = numpy.zeros((batch_size*2,128,160,1),dtype='float32')
            all_mask=numpy.zeros((batch_size*2,128,160,1),dtype='uint8')
            for mi, cur_idx in enumerate(list(idx[slice_idx])):
                cur_file_name = img_file_name[cur_idx]
                cur_uvd = uvd[cur_idx]
                bb = numpy.array([(phyMargin,phyMargin,cur_uvd[9,2])])
                bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
                margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))
                roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
                depth = numpy.asarray(roiDepth, dtype='uint16')

                axis_bounds = numpy.array([numpy.min(cur_uvd[:, 0]), numpy.max(cur_uvd[:, 0]),
                                           numpy.min(cur_uvd[:, 1]), numpy.max(cur_uvd[:, 1]),
                                           numpy.min(cur_uvd[:, 2]), numpy.max(cur_uvd[:, 2])],dtype='int32')

                tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
                tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth

                crop = tmpDepth[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                       axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth]
                loc = numpy.where(numpy.logical_and(crop>axis_bounds[4]-phyMargin,crop<axis_bounds[5]+phyMargin))
                cropmask=numpy.zeros_like(crop)

                cropmask[loc]=1
                tmpMask = numpy.zeros_like(tmpDepth,dtype='uint8')
                tmpMask[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                               axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth] =cropmask
                orimask = tmpMask[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]
                orimask = resize(orimask,(120,160), order=3,preserve_range=True)
                orimask[numpy.where(orimask>0)]=1
                all_mask[mi,4:124,:,0]=orimask
                all_input[mi,4:124,:,0]=resize(depth,(120,160), order=3,preserve_range=True)/2000.0

                jiter_width = numpy.random.randint(low=-padWidth,high=padWidth,size=1)[0]
                # jiter_width = numpy.random.randint(low=-int(padWidth/2),high=int(padWidth/2),size=1)[0]
                # print(jiter_width)

                jiter_mask = tmpMask[jiter_width+padWidth:padWidth+depth.shape[0]-jiter_width,jiter_width+padWidth:padWidth+depth.shape[1]-jiter_width]
                jiter_depth = tmpDepth[jiter_width+padWidth:padWidth+depth.shape[0]-jiter_width,jiter_width+padWidth:padWidth+depth.shape[1]-jiter_width]

                orimask = resize(jiter_mask,(120,160), order=3,preserve_range=True)
                orimask[numpy.where(orimask>0)]=1
                all_mask[mi+batch_size,4:124,:,0]=orimask
                all_input[mi+batch_size,4:124,:,0]=resize(jiter_depth,(120,160), order=3,preserve_range=True)/2000.0
                #
                # fig = plt.figure()
                # ax=fig.add_subplot(221)
                # ax.imshow(all_input[mi,4:124,:,0])
                # ax=fig.add_subplot(222)
                # ax.imshow(all_mask[mi,4:124,:,0])
                #
                # ax=fig.add_subplot(223)
                # ax.imshow(all_input[mi+batch_size,4:124,:,0])
                # ax=fig.add_subplot(224)
                # ax.imshow(all_mask[mi+batch_size,4:124,:,0])
                # plt.show()

            yield (all_input,all_mask)




def generate_fullimg_mask_from_file_unet_for_test(path,img_file_name,uvd,batch_size,n_batches):
    centerU=315.944855
    phyMargin=50.0
    padWidth=200
    img_rows=480
    img_cols=640

    num_imgs=len(img_file_name)
    idx = numpy.arange(len(batch_size*n_batches))
    num = (batch_size - num_imgs%batch_size)%batch_size
    idx = numpy.concatenate([idx,idx[0:num]],axis=0)

    x0 = numpy.zeros((batch_size*n_batches,img_rows,img_cols,1),dtype='float32')
    y = numpy.zeros((batch_size*n_batches,img_rows,img_cols,1),dtype='uint8')

    for mi, cur_idx in enumerate(list(idx)):
        cur_file_name = img_file_name[cur_idx]
        cur_uvd = uvd[cur_idx]

        bb = numpy.array([(phyMargin,phyMargin,cur_uvd[9,2])])
        bbox_uvd = xyz_uvd.xyz2uvd(setname='mega',xyz=bb)
        margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))

        roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
        depth = numpy.asarray(roiDepth, dtype='uint16')
        x0[mi,:,:,0]=depth/2000.0

        axis_bounds = numpy.array([numpy.min(cur_uvd[:, 0]), numpy.max(cur_uvd[:, 0]),
                                   numpy.min(cur_uvd[:, 1]), numpy.max(cur_uvd[:, 1]),
                                   numpy.min(cur_uvd[:, 2]), numpy.max(cur_uvd[:, 2])],dtype='int32')

        tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
        tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth

        crop = tmpDepth[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
               axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth]
        loc = numpy.where(numpy.logical_and(crop>axis_bounds[4]-phyMargin,crop<axis_bounds[5]+phyMargin))
        cropmask=numpy.zeros_like(crop)

        cropmask[loc]=1
        orimask = numpy.zeros_like(tmpDepth,dtype='uint8')
        orimask[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                       axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth] =cropmask
        y[mi,:,:,0] = orimask[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]

    return x0,y

def generate_arrays_from_file_unet(path,img_file_name,uvd,batch_size):
    output_down_ratio = 8.0
    img_rows=120
    img_cols=160
    num_imgs=len(img_file_name)
    idx = numpy.arange(len(img_file_name))
    num = (batch_size - num_imgs%batch_size)%batch_size
    idx = numpy.concatenate([idx,idx[0:num]],axis=0)
    n_batches = int(idx.shape[0]/batch_size)

    x0 = numpy.zeros((batch_size,img_rows+8,img_cols,1),dtype='float32')
    y = numpy.zeros((batch_size,int((img_rows+8)/output_down_ratio),int(img_cols/output_down_ratio),1),dtype='float32')
    # print('$'*20, 'validataion n_batches', n_batches)
    target_rows = y[0].shape[0]
    target_cols = y[1].shape[1]
    while 1:
        idx= shuffle(idx,random_state=0)
        for minibatch_index in range(n_batches):
            # print('minibatch_index',minibatch_index)
            slice_idx = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
            for mi, cur_idx in enumerate(list(idx[slice_idx])):
                cur_file_name = img_file_name[cur_idx]
                cur_uvd = uvd[cur_idx]

                roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
                roiDepth = numpy.asarray(roiDepth, dtype='uint16')/2000.0
                depth = resize(roiDepth,(img_rows,img_cols), order=3,preserve_range=True)
                u_norm = int(cur_uvd[9,0]/4/output_down_ratio)
                v_norm = int((cur_uvd[9,1]/4.0+4)/output_down_ratio)
                if v_norm>0 and u_norm >0 and v_norm<target_rows and u_norm < target_cols:
                    y[mi,v_norm,u_norm,0]=1
                x0[mi,4:(4+img_rows),:,0]=depth
            yield (x0,y)

def tmp(path,img_file_name,uvd,batch_size):
    output_down_ratio = 4.0
    img_rows=120
    img_cols=160
    num_imgs=uvd.shape[0]
    idx = numpy.arange(num_imgs)
    num = (batch_size - num_imgs%batch_size)%batch_size
    idx = numpy.concatenate([idx,idx[0:num]],axis=0)
    n_batches = int(idx.shape[0]/batch_size)

    x0 = numpy.zeros((batch_size,img_rows+8,img_cols,1),dtype='float32')
    y = numpy.zeros((batch_size,int((img_rows+8)/output_down_ratio),int(img_cols/output_down_ratio),1),dtype='float32')
    # print('$'*20, 'validataion n_batches', n_batches)

    idx= shuffle(idx,random_state=0)
    for minibatch_index in range(n_batches):
        # print minibatch_index
        slice_idx = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
        for mi, cur_idx in enumerate(list(idx[slice_idx])):
            cur_file_name = img_file_name[cur_idx]
            cur_uvd = uvd[cur_idx]

            roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
            roiDepth = numpy.asarray(roiDepth, dtype='uint16')/2000.0
            print(numpy.max(roiDepth))
            depth = resize(roiDepth,(img_rows,img_cols), order=3,preserve_range=True)
            u_norm = int(cur_uvd[9,0]/16)
            v_norm = int((cur_uvd[9,1]/4.0+4)/4.0)
            y[mi,v_norm,u_norm,0]=1

            x0[mi,4:(4+img_rows),:,0]=depth
            plt.imshow(x0[mi,:,:,0],'gray')
            plt.figure()
            tmp = resize(x0[mi,:,:,0],(y[0].shape[0],y[0].shape[1]), order=3,preserve_range=True)
            plt.imshow(tmp,'gray')
            plt.scatter(u_norm,v_norm)
            plt.show()
        # print('yield validataion minibatch_index ',minibatch_index)


def generate_fullimg_mask_from_file_unet_show(path,img_file_name,uvd,batch_size):
    centerU=315.944855
    phyMargin=50.0
    padWidth=200
    img_rows=480
    img_cols=640

    num_imgs=len(img_file_name)
    idx = numpy.arange(len(img_file_name))
    num = (batch_size - num_imgs%batch_size)%batch_size
    idx = numpy.concatenate([idx,idx[0:num]],axis=0)
    n_batches = int(idx.shape[0]/batch_size)


    idx= shuffle(idx,random_state=0)
    for minibatch_index in range(n_batches):
        # print('minibatch_index',minibatch_index)
        slice_idx = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
        for mi, cur_idx in enumerate(list(idx[slice_idx])):
            cur_file_name = img_file_name[cur_idx]
            cur_uvd = uvd[cur_idx]

            bb = numpy.array([(phyMargin,phyMargin,cur_uvd[9,2])])
            bbox_uvd = xyz_uvd.xyz2uvd(setname='mega',xyz=bb)
            margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))

            roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
            depth = numpy.asarray(roiDepth, dtype='uint16')


            axis_bounds = numpy.array([numpy.min(cur_uvd[:, 0]), numpy.max(cur_uvd[:, 0]),
                                       numpy.min(cur_uvd[:, 1]), numpy.max(cur_uvd[:, 1]),
                                       numpy.min(cur_uvd[:, 2]), numpy.max(cur_uvd[:, 2])],dtype='int32')

            tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
            tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth

            crop = tmpDepth[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                   axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth]
            loc = numpy.where(numpy.logical_and(crop>axis_bounds[4]-phyMargin,crop<axis_bounds[5]+phyMargin))
            cropmask=numpy.zeros_like(crop)

            cropmask[loc]=1
            orimask = numpy.zeros_like(tmpDepth,dtype='uint8')
            orimask[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                           axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth] =cropmask
            mask= orimask[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]

            plt.figure()
            plt.imshow(depth,'gray')
            plt.figure()
            plt.imshow(mask,'gray')
            plt.show()

def generate_train_tmp(path,img_file_name,uvd,batch_size):
    num_imgs=uvd.shape[0]
    idx = numpy.arange(num_imgs)

    print(idx.shape)
    n_batches=int(idx.shape[0]/batch_size)


    phyMargin=50.0
    padWidth=200
    centerU=315.944855
    setname='mega'


    idx= shuffle(idx,random_state=0)
    for minibatch_index in range(n_batches):
        # print('minibatch_index',minibatch_index)
        slice_idx = range(minibatch_index * batch_size, (minibatch_index + 1) * batch_size,1)
        all_input = numpy.zeros((batch_size,128,160,1),dtype='float32')
        all_mask=numpy.zeros((batch_size,128,160,1),dtype='uint8')

        for mi, cur_idx in enumerate(list(idx[slice_idx])):
            cur_file_name = img_file_name[cur_idx]
            cur_uvd = uvd[cur_idx]
            bb = numpy.array([(phyMargin,phyMargin,cur_uvd[9,2])])
            bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
            margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))
            roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
            depth = numpy.asarray(roiDepth, dtype='uint16')

            axis_bounds = numpy.array([numpy.min(cur_uvd[:, 0]), numpy.max(cur_uvd[:, 0]),
                                       numpy.min(cur_uvd[:, 1]), numpy.max(cur_uvd[:, 1]),
                                       numpy.min(cur_uvd[:, 2]), numpy.max(cur_uvd[:, 2])],dtype='int32')

            tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
            tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth

            crop = tmpDepth[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                   axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth]
            loc = numpy.where(numpy.logical_and(crop>axis_bounds[4]-phyMargin,crop<axis_bounds[5]+phyMargin))
            cropmask=numpy.zeros_like(crop)

            cropmask[loc]=1
            orimask = numpy.zeros_like(tmpDepth,dtype='uint8')
            orimask[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
                           axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth] =cropmask
            orimask = orimask[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]
            orimask = resize(orimask,(120,160), order=3,preserve_range=True)
            orimask[numpy.where(orimask>0)]=1
            all_mask[mi,4:124,:,0]=orimask
            all_input[mi,4:124,:,0]=resize(depth,(120,160), order=3,preserve_range=True)/2000.0

            plt.figure()
            plt.imshow(depth,'gray')
            plt.scatter(cur_uvd[:,0],cur_uvd[:,1])
            plt.figure()
            plt.imshow(orimask,'gray')
            plt.figure()
            plt.imshow(crop,'gray')
            plt.show()

        for i in range(batch_size):
            plt.figure()
            plt.imshow(all_input[i,:,:,0],'gray')
            plt.figure()
            plt.imshow(all_mask[i,:,:,0],'gray')
            plt.show()

import h5py
if __name__ == '__main__':
    source_dir = 'F:/BigHand_Challenge/Training/'
    save_dir = 'F:/HuaweiProj/data/mega'

    f = h5py.File('%s/source/test_mask.h5'%save_dir, 'r')
    filename = f['filename'][...]
    uvd = f['uvd'][...]
    f.close()
    generate_train_tmp(path=source_dir,img_file_name=filename,uvd=uvd,batch_size=32)
#     base_dir = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/NYU_dataset/NYU_dataset/'
#     generate_arrays_from_file_unet(path=base_dir,
#             dataset='test',num_imgs=8252,num_classes=17,batch_size=1)
#     # create_data(dataset='train',num_imgs = 72757)
#     # create_data(dataset='test',num_imgs = 8252)
#     uvd_jnt_gt,_,file_name=get_filenames_labels(dataset_dir=source_dir)
#     num_img=len(file_name)
#     idx = shuffle(numpy.arange(num_img),random_state=0)
#     img_idx_train = idx[:int(num_img*0.9)]
#     img_idx_test = idx[int(num_img*0.9):]
#     generate_train_tmp(path=source_dir,img_file_name=file_name[img_idx_test],uvd=uvd_jnt_gt[img_idx_test],batch_size=32)
#
#     generate_fullimg_mask_from_file_unet_show(path=source_dir,img_file_name=file_name[img_idx_test],uvd=uvd_jnt_gt[img_idx_test],batch_size=32)