from __future__ import print_function
from sklearn.utils import shuffle
from PIL import Image
import numpy
import matplotlib.pyplot as plt
from ..utils import xyz_uvd
from skimage.transform import resize
import h5py

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
    return uvd_jnt_gt,xyz_jnt_gt,numpy.array(file_name,dtype=object)

def craete_data_for_heatmap(path,img_file_name,uvd,batch_size):
    output_down_ratio = 4.0
    img_rows=120
    img_cols=160
    num_imgs=uvd.shape[0]
    idx = numpy.arange(num_imgs)
    num = (batch_size - num_imgs%batch_size)%batch_size
    idx = numpy.concatenate([idx,idx[0:num]],axis=0)

    x0 = numpy.zeros((idx.shape[0],img_rows+8,img_cols,1),dtype='float32')
    y = numpy.zeros((idx.shape[0],int((img_rows+8)/output_down_ratio),int(img_cols/output_down_ratio),1),dtype='float32')

    target_rows = y.shape[1]
    target_cols = y.shape[2]

    for mi, cur_idx in enumerate(list(idx)):
        print(mi,'/',idx.shape[0])
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

        # plt.figure()
        # plt.imshow(x0[mi,:,:,0],'gray')
        # plt.figure()
        # plt.imshow(y[mi,:,:,0],'gray')
        # plt.figure()
        # tmp = resize(x0[mi,:,:,0],(y[0].shape[0],y[0].shape[1]), order=3,preserve_range=True)
        # plt.imshow(tmp,'gray')
        # plt.scatter(u_norm,v_norm)
        # plt.show()
        # print('yield validataion minibatch_index ',minibatch_index)
    f = h5py.File('%s/test_detector.h5'%save_dir, 'w')
    f.create_dataset('x', data=x0)
    f.create_dataset('y', data=y)
    f.close()



def hand_mask(path,img_file_name,uvd,batch_size):
    img_rows=120
    img_cols=160
    num_imgs=uvd.shape[0]
    idx = numpy.arange(num_imgs)
    num = (batch_size - num_imgs%batch_size)%batch_size
    idx = numpy.concatenate([idx,idx[0:num]],axis=0)

    phyMargin=30.0
    padWidth=200
    centerU=315.944855
    setname='mega'

    for mi, cur_idx in enumerate(list(idx)):
        print(mi,'/',idx.shape[0])
        cur_file_name = img_file_name[cur_idx]
        cur_uvd = uvd[cur_idx]
        bb = numpy.array([(phyMargin,phyMargin,cur_uvd[9,2])])
        bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
        print(bbox_uvd)
        margin = int(numpy.ceil(bbox_uvd[0,0] - centerU)/4.0)
        print(margin)


        roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
        roiDepth = numpy.asarray(roiDepth, dtype='uint16')
        depth = resize(roiDepth,(img_rows,img_cols), order=3,preserve_range=True)

        cur_uvd[:,0:2] /=4.0
        # cur_uvd[:,1]+=4.0

        axis_bounds = numpy.array([numpy.min(cur_uvd[:, 0]), numpy.max(cur_uvd[:, 0]),
                                   numpy.min(cur_uvd[:, 1]), numpy.max(cur_uvd[:, 1]),
                                   numpy.min(cur_uvd[:, 2]), numpy.max(cur_uvd[:, 2])],dtype='int32')
        print(axis_bounds)

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

        plt.figure()
        plt.imshow(depth,'gray')
        plt.figure()
        plt.imshow(crop,'gray')
        plt.figure()
        plt.imshow(orimask,'gray')
        plt.show()


def hand_mask_ori(path,img_file_name,uvd,xyz,batch_size,dataset):

    num_imgs=uvd.shape[0]
    idx = numpy.arange(num_imgs)
    num = (batch_size - num_imgs%batch_size)%batch_size
    idx = numpy.concatenate([idx,idx[0:num]],axis=0)
    print(idx.shape)

    phyMargin=50.0
    padWidth=200
    centerU=315.944855
    setname='mega'
    all_input = numpy.empty((idx.shape[0],120,160),dtype='float32')
    all_mask=numpy.empty((idx.shape[0],120,160),dtype='uint8')


    # for mi, cur_idx in enumerate(list(idx)):
    #     print(mi,'/',idx.shape[0])
    #     cur_file_name = img_file_name[cur_idx]
    #     cur_uvd = uvd[cur_idx]
    #     bb = numpy.array([(phyMargin,phyMargin,cur_uvd[9,2])])
    #     bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
    #     margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))
    #     roiDepth = Image.open('%s/images/%s' %(path,cur_file_name))
    #     depth = numpy.asarray(roiDepth, dtype='uint16')
    #
    #     axis_bounds = numpy.array([numpy.min(cur_uvd[:, 0]), numpy.max(cur_uvd[:, 0]),
    #                                numpy.min(cur_uvd[:, 1]), numpy.max(cur_uvd[:, 1]),
    #                                numpy.min(cur_uvd[:, 2]), numpy.max(cur_uvd[:, 2])],dtype='int32')
    #
    #     tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
    #     tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth
    #
    #     crop = tmpDepth[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
    #            axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth]
    #     loc = numpy.where(numpy.logical_and(crop>axis_bounds[4]-phyMargin,crop<axis_bounds[5]+phyMargin))
    #     cropmask=numpy.zeros_like(crop)
    #
    #     cropmask[loc]=1
    #     orimask = numpy.zeros_like(tmpDepth,dtype='uint8')
    #     orimask[axis_bounds[2]-margin+padWidth:axis_bounds[3]+margin+padWidth,
    #                    axis_bounds[0]-margin+padWidth:axis_bounds[1]+margin+padWidth] =cropmask
    #     orimask = orimask[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]
    #     orimask = resize(orimask,(120,160), order=3,preserve_range=True)
    #     orimask[numpy.where(orimask>0)]=1
    #     all_mask[mi]=orimask
    #     all_input[mi]=resize(depth,(120,160), order=3,preserve_range=True)
    #
    #     plt.figure()
    #     plt.imshow(depth,'gray')
    #     plt.scatter(cur_uvd[:,0],cur_uvd[:,1])
    #     plt.figure()
    #     plt.imshow(all_input[mi],'gray')
    #     plt.figure()
    #     plt.imshow(crop,'gray')
    #     plt.figure()
    #     plt.imshow(all_mask[mi],'gray')
    #     plt.show()
    #
    # f = h5py.File('%s/%s_mask.h5'%(save_dir,dataset), 'w')
    # f.create_dataset('x', data=all_input)
    # f.create_dataset('mask', data=all_mask)
    # f.create_dataset('uvd', data=uvd[idx])
    # f.create_dataset('xyz', data=xyz[idx])
    # dt = h5py.special_dtype(vlen=str)
    # f.create_dataset('filename', data=img_file_name[idx],dtype=dt)
    # f.close()

    f = h5py.File('%s/%s_mask.h5'%(save_dir,dataset), 'r+')
    del f['uvd']
    del f['xyz']
    del f['filename']
    f.create_dataset('uvd', data=uvd[idx])
    f.create_dataset('xyz', data=xyz[idx])
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset('filename', data=img_file_name[idx],dtype=dt)
    f.close()


if __name__ == '__main__':
    # source_dir = '/media/Data/Qi/data/BigHand_Challenge/Training/'
    # save_dir = '/media/Data/Qi/data/source'
    #
    source_dir = 'F:/BigHand_Challenge/Training'
    save_dir = 'F:/HuaweiProj/data/mega/source'

    uvd_jnt_gt,xyz,file_name=get_filenames_labels(dataset_dir=source_dir)
    num_img=len(file_name)
    idx = shuffle(numpy.arange(num_img),random_state=0)
    img_idx_train = idx[:int(num_img*0.9)]
    img_idx_test = idx[int(num_img*0.9):]
    print(img_idx_train.shape)
    hand_mask_ori(path=source_dir,img_file_name=file_name[img_idx_test],uvd=uvd_jnt_gt[img_idx_test],xyz=xyz[img_idx_test],batch_size=128,dataset='test')