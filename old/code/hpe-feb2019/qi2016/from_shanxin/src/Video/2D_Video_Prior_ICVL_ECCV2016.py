__author__ = 'QiYE'
import h5py
import numpy
import scipy.io
from src.utils import constants
import matplotlib.pyplot as plt
from src.utils import xyz_result_path

import matplotlib.gridspec as gridspec

from src.utils.xyz_uvd import  xyz2uvd

def subplot_row(fig,depth,gs,i_gs,xyz_true,uvd_hso,uvd_hybrid):
    linewidth=3
    markersize=10
    ax_wid = 71
    ax = fig.add_subplot(gs[i_gs])
    ax.imshow(depth,'gray')
    dot = xyz_true

    for k in [1,5,9,13,17]:

        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]

        ax.plot(x,y,linewidth=linewidth,marker='o',markersize=markersize,c='m')

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]

        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)

    ax.set_xlim(0, ax_wid)
    ax.set_ylim(ax_wid,0 )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+1])
    ax.imshow(depth,'gray')
    dot = uvd_hso

    for k in [1,5,9,13,17]:

        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        ax.plot(x,y,c='m',linewidth=linewidth,marker='o',markersize=markersize)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)

    ax.set_xlim(0, ax_wid)
    ax.set_ylim(ax_wid,0 )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+2])
    ax.imshow(depth,'gray')
    dot = uvd_hybrid


    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        ax.plot(x,y,linewidth=linewidth,marker='o',c='m',markersize=markersize)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)


    ax.set_xlim(0, ax_wid)
    ax.set_ylim(ax_wid,0 )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()


if __name__ =='__main__':


    dataset_path_prefix=constants.Data_Path
    setname='ICVL'
    dataset ='test'

    source_name=xyz_result_path.icvl_source_name_ori


    xyz_hybrid =numpy.empty((1596,21,3),dtype='float32')
    for i in xrange(21):
        xyz_hybrid[:,i,:] = scipy.io.loadmat('C:/Proj/Proj_CNN_Hier/data/HDJIF_cmp_prior/ICVL Final Result 1117/jnt%d_xyz.mat'%i)['jnt']
    xyz_hso = scipy.io.loadmat('C:/Users/QiYE/OneDrive/Doc_ProgressReport/eccv2016/images/%s_hso_slice'%setname)['xyz']

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    print xyz_true.shape
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_uvd_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    uvd_true = keypoints['uvd']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']


    idx =[0,
          1,6,11,16,
          2,7,12,17,
          3,8,13,18,
          4,9,14,19,
          5,10,15,20]

    dataset_dir =  xyz_result_path.msrc_raw_dataset_dir
    kinect_index = 1


    src_path='%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    r0=f['r0'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    derot_uvd = f['joint_label_uvd'][...]
    f.close()


    cmap = plt.cm.rainbow
    colors_map = cmap(numpy.arange(cmap.N))
    rng = numpy.random.RandomState(0)
    num = rng.randint(0,256,(21,))
    jnt_colors = colors_map[num]
    print jnt_colors.shape
    #3,13
    rng1 = numpy.random.RandomState(13)

    for image_index in xrange(100,150,1):
        print 'image_index',image_index
        fig = plt.figure(figsize=(16, 12), facecolor='w')
        # fig = plt.figure(figsize=(9, 4))
        ax0 = fig.add_axes([0., 0., 1., 1., ], axisbg='w')
        ax0.text(0.5,0.85,'%s Test Sequence '%setname,ha='center', va='bottom',size=30,color='k')
        ax0.text(0.9,0.25,'Frame Number: %d'%(image_index+1),ha='right', va='center',size=20,color='k')
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.set_axis_off()
        gs = gridspec.GridSpec(1,3,left=0.05, right=0.95,bottom=0.4,top=0.7,wspace=0.0, hspace=0)

        v0_off=0.75
        step=1/3.0
        h0_off = step/2
        plt.text(h0_off,v0_off,'GroundTruth',ha='center', va='center',size=20,color='k')
        plt.text(h0_off+step,v0_off,'HSO',ha='center', va='center',size=20,color='k')
        plt.text(h0_off+step*2,v0_off,'Ours',ha='center', va='center',size=20,color='k')

        v0_off=0.35
        plt.text(h0_off,v0_off,'Errors: ',ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_hso[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+step*1,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_hybrid[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+step*2,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')


        i=image_index

        vmin=rect_d1d2w[i,0]
        umin =rect_d1d2w[i,1]
        urange=rect_d1d2w[i,2]

        uvd_true = xyz2uvd(setname,xyz_true[i],roixy,jnt_type='single')
        uvd_hso= xyz2uvd(setname,xyz_hso[i],roixy,jnt_type='single')
        uvd_hybrid= xyz2uvd(setname,xyz_hybrid[i],roixy,jnt_type='single')


        hand_wid = 72
        uvd_true[:, 0] = (uvd_true[:, 0] - umin )/urange*hand_wid+12
        uvd_true[:, 1]=(uvd_true[:, 1] - vmin )/urange*hand_wid+12

        uvd_hso[:, 0] = (uvd_hso[:, 0] - umin )/urange*hand_wid+12
        uvd_hso[:, 1]= (uvd_hso[:, 1] - vmin )/urange*hand_wid+12

        uvd_hybrid[:, 0]= (uvd_hybrid[:, 0] - umin )/urange*hand_wid+12
        uvd_hybrid[:, 1]=(uvd_hybrid[:, 1] - vmin )/urange*hand_wid+12

        i_gs=0
        subplot_row(fig,r0[i,12:84,12:84],gs,i_gs,uvd_true,uvd_hso,uvd_hybrid)



        # plt.show()
        # plt.savefig('C:/Proj/Proj_CNN_Hier/data/HDJIF_cmp_prior/%s_ECCV2016_EXAMP/%s%04d.eps'%(setname,setname,image_index+1),format='eps',dpi=300)
        plt.savefig('C:/Proj/Proj_CNN_Hier/data/HDJIF_cmp_prior/%s_CMP_VIDEO_ECCV2016/%sCMP%04d.png'%(setname,setname,image_index),format='png',dpi=300)
        plt.close('all')

        # plt.savefig('C:/users/QiYE/OneDrive/Doc_ProgressReport/iros2016/%s/%d.png'%(setname,image_index),format='png', dpi=1200,bbox_inches='tight',frameon=False,face_color='k',pad_inches=0.1)