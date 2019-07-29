__author__ = 'QiYE'
import matplotlib
matplotlib.use('Agg')
import h5py
from PIL import Image
import numpy
from skimage.transform import resize
import matplotlib.pyplot as plt
from  ..utils import xyz_uvd,hand_utils
from mpl_toolkits.mplot3d import Axes3D


from ..utils import math,loss,show_blend_img
from keras.models import Model,model_from_json
from keras.optimizers import Adam
import os
import tensorflow as tf



os.environ["CUDA_VISIBLE_DEVICES"]="3"
import keras.backend.tensorflow_backend as KTF
def get_session(gpu_fraction=0.2):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())


cmap = plt.cm.rainbow
colors_map = cmap(numpy.arange(cmap.N))
rng = numpy.random.RandomState(0)
num = rng.randint(0,256,(21,))
jnt_colors = colors_map[num]
# print jnt_colors.shape
markersize = 7
linewidth=2
azim =  -177
elev = -177

hand_img_size=96
hand_size=300.0
centerU=315.944855
padWidth=100

def show_hand_skeleton(refSkeleton,centre):
    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = refSkeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='b',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='b',linewidth=linewidth,marker='o',markersize=markersize+3)
    ax.plot(centre[:,2],centre[:,0],centre[:,1],c='r',linewidth=linewidth,marker='o',markersize=markersize+10)
    # ax.set_xlim(axis_bounds[5], axis_bounds[4])
    # ax.set_ylim(axis_bounds[0], axis_bounds[1])
    # ax.set_zlim(axis_bounds[2], axis_bounds[3])
    # ax.azim = azim
    # ax.elev = elev
    # # ax.set_autoscale_on(True)
    # ax.grid(True)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # ax.set_axis_off()
    plt.show()



def getLabelformMsrcFormat(base_path,dataset):

    xyz_jnt_gt=[]
    file_name = []
    our_index = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
    with open('%s/%s_annotation.txt'%(base_path,dataset), mode='r',encoding='utf-8',newline='') as f:
        for line in f:
            part = line.split('\t')
            # print(part)
            file_name.append(part[0])
            xyz_jnt_gt.append(part[1:64])
    f.close()

    xyz_jnt_gt=numpy.array(xyz_jnt_gt,dtype='float32')
    print(xyz_jnt_gt.shape)

    xyz_jnt_gt.shape=(xyz_jnt_gt.shape[0],21,3)
    xyz_jnt_gt=xyz_jnt_gt[:,our_index,:]
    uvd_jnt_gt =xyz_uvd.xyz2uvd(xyz=xyz_jnt_gt,setname='mega')

    return uvd_jnt_gt,xyz_jnt_gt,file_name



def load_model(save_dir,version):
    print(version)
    # load json and create model
    json_file = open("%s/detector/best/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/detector/best/weight_%s.h5"%(save_dir,version))
    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=loss.cost_sigmoid)
    return loaded_model


def get_mask(depthimg,model):

    depth=numpy.zeros((1,480,640,1))
    depth[0,:,:,0] = depthimg/2000.0
    mask = model.predict(x=depth,batch_size=1)
    # out =numpy.zeros(mask.shape,dtype='uint8')
    mask = math.sigmoid(mask[0,:,:,0])
    # out[numpy.where(mask>0.5)]=
    return mask

def prepare_data():

    datasets=['test']

    setname='mega'
    # model_save_dir ='/media/Data/Qi/data'
    # img_dir = '/media/Data/shanxin/megahand/'
    # anno_dir='/media/Data/Qi/data/train_test_annotation/'
    # save_path='/media/Data/Qi/data/source'

    model_save_dir ='F:/HuaweiProj/data/mega'
    img_dir = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/MegaEgo/'
    anno_dir='F:/HuaweiProj/data/mega/train_test_annotation/'
    save_path='F:/HuaweiProj/data/mega/source'

    for dataset in datasets:
        print(dataset)

        imgs0=[]
        imgs1=[]
        imgs2=[]
        uvd_gt=[]
        uvd_centre=[]
        uvd_norm_gt=[]
        new_file_names=[]

        xyz_gt=[]

        uvd, xyz_joint,file_name =getLabelformMsrcFormat(base_path=anno_dir,dataset=dataset)
        model= load_model(save_dir=model_save_dir,version='pixel_fullimg_ker32_lr0.001000')

        # for i in numpy.random.randint(0,len(file_name),30):
        for i in range(len(file_name)):
            print(i)
            uvd_joint_gt=uvd[i]
            xyz_joint_gt=xyz_joint[i]
            cur_frame = file_name[i]

            depth = Image.open("%s%s.png"%(img_dir,cur_frame))
            depth = numpy.asarray(depth, dtype='uint16')
            print(depth.shape,depth.max(),depth.min(),depth.dtype)
            mask = get_mask(depthimg=depth,model=model)

            # fig = plt.figure()
            # plt.imshow(depth,'gray')
            # plt.show()

            #
            backimg,colors, cmap = show_blend_img.show_two_imgs(backimg=depth,topimg=mask,alpha=0.2)
            fig, ax = plt.subplots()
            ax.imshow(backimg,'gray')
            ax.imshow(colors,cmap=cmap)
            plt.scatter(uvd_joint_gt[:,0],uvd_joint_gt[:,1])
            plt.show()


            loc = numpy.where(mask>0.5)

            if  loc[0].shape[0]<30:
                print('no hand in the area ',cur_frame)
                continue
            depth_value = depth[loc]
            # print('loc',loc[0].shape,numpy.max(loc[1]),numpy.min(loc[1]),numpy.max(loc[1]),numpy.min(loc[1]),',depth_value',depth_value,)
            U = numpy.mean(loc[1])
            V = numpy.mean(loc[0])
            D = numpy.mean(depth_value)
            if D<10:
                print('not valid hand area',cur_frame)
                continue
            meanUVD = numpy.array([(U,V,D)])
            meanXYZ=xyz_uvd.uvd2xyz(setname=setname,uvd=meanUVD )
            gtMean = numpy.mean(xyz_joint_gt,axis=0)
            dist =   numpy.sqrt(numpy.sum((gtMean-meanXYZ)**2,axis=-1))
            # print(dist)
            if dist>hand_size/3:
                print('wrong detection',cur_frame)
                backimg,colors, cmap = show_blend_img.show_two_imgs(backimg=depth,topimg=mask,alpha=0.2)
                fig, ax = plt.subplots()
                ax.imshow(backimg,'gray')
                ax.imshow(colors,cmap=cmap)
                plt.scatter(uvd_joint_gt[:,0],uvd_joint_gt[:,1])
                plt.savefig('%s/wrong_detection/%s_%d.png'%(save_path,dataset,i))
                plt.close()
                # plt.show()

                continue

            bb = numpy.array([(hand_size,hand_size,numpy.mean(depth_value))])
            bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
            margin = int(numpy.ceil(bbox_uvd[0,0] - centerU))

            depth_w_hand_only = depth.copy()
            loc_back = numpy.where(mask<0.5)
            depth_w_hand_only[loc_back]=0
            loc_back = numpy.where(numpy.logical_and(depth_w_hand_only>D+hand_size/2,depth_w_hand_only<D-hand_size/2))
            depth_w_hand_only[loc_back]=0

            tmpDepth = numpy.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
            tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]=depth_w_hand_only
            if U-margin/2+padWidth<0 or U+margin/2+padWidth>tmpDepth.shape[1]-1 or V - margin/2+padWidth <0 or V+margin/2+padWidth>tmpDepth.shape[0]-1:
                print('hand part outside the image',cur_frame )
                continue
            crop = tmpDepth[int(V-margin/2+padWidth):int(V+margin/2+padWidth),int(U-margin/2+padWidth):int(U+margin/2+padWidth)]

            norm_hand_img=numpy.ones(crop.shape,dtype='float32')
            loc_hand=numpy.where(crop>0)
            norm_hand_img[loc_hand]=(crop[loc_hand]-D)/hand_size
            # plt.figure()
            # plt.imshow(norm_hand_img,'gray')
            # plt.show()


            r0 = resize(norm_hand_img, (hand_img_size,hand_img_size), order=3,preserve_range=True)
            r1 = resize(norm_hand_img, (hand_img_size/2,hand_img_size/2), order=3,preserve_range=True)
            r2 = resize(norm_hand_img, (hand_img_size/4,hand_img_size/4), order=3,preserve_range=True)


            uvd_norm = uvd_joint_gt.copy()
            uvd_norm[:,0] = (uvd_norm[:,0]-U)/margin
            uvd_norm[:,1]=(uvd_norm[:,1]-V)/margin
            uvd_norm[:,2]=(uvd_norm[:,2]-D)/hand_size

            # plt.figure()
            # plt.imshow(r0,'gray')
            # plt.scatter(uvd_norm[:,0]*hand_img_size+48,uvd_norm[:,1]*hand_img_size+48)
            # # plt.savefig('%s/norm_hand/%s_%d.png'%(save_path,dataset,i))
            # plt.show()

            imgs0.append(r0)
            imgs1.append(r1)
            imgs2.append(r2)

            xyz_gt.append(xyz_joint_gt)

            uvd_gt.append(uvd_joint_gt)
            uvd_centre.append([U,V,D])
            uvd_norm_gt.append(uvd_norm)
            new_file_names.append(cur_frame)
            # show_hand_skeleton(xyz_joint_gt,meanXYZ)


        # f = h5py.File('%s/%s_crop_norm.h5'%(save_path,dataset), 'w')
        # new_file_names=numpy.array(new_file_names,dtype=object)
        # dt = h5py.special_dtype(vlen=str)
        # f.create_dataset('new_file_names', data=new_file_names,dtype=dt)
        # f.create_dataset('img0', data=imgs0)
        # f.create_dataset('img1', data=imgs1)
        # f.create_dataset('img2', data=imgs2)
        # f.create_dataset('xyz_gt', data=xyz_gt)
        # f.create_dataset('uvd_gt', data=uvd_gt)
        # f.create_dataset('uvd_hand_centre', data=uvd_centre)
        # f.create_dataset('uvd_norm_gt', data=uvd_norm_gt)
        #
        # f.create_dataset('bbsize', data=hand_size)
        # f.close()
def readming():

    datasets=['test']

    setname='mega'
    # model_save_dir ='/media/Data/Qi/data'
    # img_dir = '/media/Data/shanxin/megahand/'
    # anno_dir='/media/Data/Qi/data/train_test_annotation/'
    # save_path='/media/Data/Qi/data/source'

    model_save_dir ='F:/HuaweiProj/data/mega'
    img_dir = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/MegaEgo/'
    anno_dir='F:/HuaweiProj/data/mega/train_test_annotation/'
    save_path='F:/HuaweiProj/data/mega/source'



    for dataset in datasets:
        print(dataset)


        uvd, xyz_joint,file_name =getLabelformMsrcFormat(base_path=anno_dir,dataset=dataset)


        for i in numpy.random.randint(0,len(file_name),300):
        # for i in range(len(file_name)):
            print(i)
            uvd_joint_gt=uvd[i]
            xyz_joint_gt=xyz_joint[i]
            cur_frame = file_name[i]

            depth = Image.open("%s%s.png"%(img_dir,cur_frame))
            depth = numpy.asarray(depth, dtype='uint16')
            print(depth.shape,depth.max(),depth.min(),depth.dtype)

if __name__=='__main__':
    readming()
    # prepare_data()


