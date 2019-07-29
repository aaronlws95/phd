__author__ = 'QiYE'
import h5py
import numpy
import scipy.io
from src.utils import constants
import matplotlib.pyplot as plt
from src.utils import xyz_result_path

import matplotlib.gridspec as gridspec

from src.utils.xyz_uvd import  xyz2uvd

def subplot_row(fig,depth,gs,i_gs,xyz_true,xyz_pred_hol,xyz_pred_hol_derot,xyz_pred_holi_derot,uvd_hier_recur,uvd_hybrid):
    linewidth=2
    markersize=7
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
    dot = uvd_hier_recur

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

    ax = fig.add_subplot(gs[i_gs+3])
    ax.imshow(depth,'gray')
    dot = xyz_pred_hol

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

    ax = fig.add_subplot(gs[i_gs+4])
    ax.imshow(depth,'gray')
    dot = xyz_pred_hol_derot

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

    ax = fig.add_subplot(gs[i_gs+5])
    ax.imshow(depth,'gray')
    dot = xyz_pred_holi_derot

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
    setname='NYU'
    dataset ='test'



    source_name=xyz_result_path.nyu_source_name_raw
    derot_whlimg=xyz_result_path.nyu_derot_whlimg
    hol_path = xyz_result_path.nyu_hol_path_raw
    hol_derot_path = xyz_result_path.nyu_hol_derot_path_raw
    slice_range = 900

    xyz_pred_holi_derot=numpy.empty((slice_range,21,3),dtype='float32')
    for i in xrange(21):
        file=scipy.io.loadmat('C:/Proj/Proj_CNN_Hier/data/nyu/nyu_holi_derot_recur_8252/jnt%d_xyz.mat'%i)
        xyz_pred_holi_derot[:,i,:]=file['jnt'][0:slice_range]

    xyz_hier_recur=numpy.empty((slice_range,21,3),dtype='float32')
    for i in xrange(21):
        file=scipy.io.loadmat('C:/Proj/Proj_CNN_Hier/data/nyu/nyu_hier_derot_recur_8252/jnt%d_xyz.mat'%i)
        xyz_hier_recur[:,i,:]=file['jnt'][0:slice_range]


    xyz_hybrid =numpy.empty((slice_range,21,3),dtype='float32')



    for i in xrange(21):
        xyz_hybrid[:,i,:] = scipy.io.loadmat('C:/Proj/Proj_CNN_Hier/data/nyu/nyu_cnn_pso_1000/jnt%d_xyz.mat'%i)['jnt'][0:slice_range]

    xyz_hso = scipy.io.loadmat('C:/Proj/Proj_CNN_Hier/data/HDJIF_cmp_prior/Result_Prior_Cmp/%s_hso_8252_21jnts'%setname)['xyz'][0:slice_range]

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz'][0:slice_range]
    print xyz_true.shape
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy'][0:slice_range]

    xyz_pred_hol= numpy.load('%sdata/%s/whole/best/%s.npy' % (dataset_path_prefix,setname,hol_path))[0][0:slice_range]
    print xyz_pred_hol.shape

    xyz_pred_hol_derot= numpy.load('%sdata/%s/whole_derot/best/%s.npy' % (dataset_path_prefix,setname,hol_derot_path))[0][0:slice_range]
    print xyz_pred_hol_derot.shape

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
    r0=f['r0'][...][0:slice_range]
    rect_d1d2w=f['bbox'][...][0:slice_range]
    depth_dmin_dmax = f['depth_dmin_dmax'][...][0:slice_range]
    orig_pad_border=f['orig_pad_border'][...]
    derot_uvd = f['joint_label_uvd'][...][0:slice_range]
    f.close()


    cmap = plt.cm.rainbow
    colors_map = cmap(numpy.arange(cmap.N))
    rng = numpy.random.RandomState(0)
    num = rng.randint(0,256,(21,))
    jnt_colors = colors_map[num]
    print jnt_colors.shape
    #3,13
    rng1 = numpy.random.RandomState(13)
    for image_index in xrange(100,190,3):
        print 'image_index',image_index
        fig = plt.figure(figsize=(16, 12), facecolor='w')
        # fig = plt.figure(figsize=(9, 4))
        ax0 = fig.add_axes([0., 0., 1., 1., ], axisbg='w')
        ax0.text(0.5,0.9,'%s Test Sequence '%setname,ha='center', va='bottom',size=30,color='k')
        ax0.text(0.9,0.03,'Frame Number: %d'%(image_index+1),ha='right', va='center',size=20,color='k')
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.set_axis_off()
        gs = gridspec.GridSpec(2,3,left=0.25, right=0.85,bottom=0.1,top=0.8,wspace=0.0, hspace=0)


        i=image_index


        vmin=rect_d1d2w[i,0]
        umin =rect_d1d2w[i,1]
        urange=rect_d1d2w[i,2]

        uvd_true = xyz2uvd(setname,xyz_true[i],roixy,jnt_type='single')
        uvd_hol = xyz2uvd(setname,xyz_pred_hol[i],roixy,jnt_type='single')
        uvd_hol_derot = xyz2uvd(setname,xyz_pred_hol_derot[i],roixy,jnt_type='single')
        uvd_holi_derot= xyz2uvd(setname,xyz_pred_holi_derot[i],roixy,jnt_type='single')
        uvd_hier_recur= xyz2uvd(setname,xyz_hier_recur[i],roixy,jnt_type='single')
        uvd_hybrid= xyz2uvd(setname,xyz_hybrid[i],roixy,jnt_type='single')


        hand_wid = 72
        uvd_true[:, 0] = (uvd_true[:, 0] - umin )/urange*hand_wid+12
        uvd_true[:, 1]=(uvd_true[:, 1] - vmin )/urange*hand_wid+12

        uvd_hol[ :, 0] = (uvd_hol[ :, 0] - umin )/urange*hand_wid+12
        uvd_hol[:, 1]=(uvd_hol[:, 1] - vmin )/urange*hand_wid+12

        uvd_hol_derot[:, 0] = (uvd_hol_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_hol_derot[:, 1]= (uvd_hol_derot[:, 1] - vmin )/urange*hand_wid+12

        uvd_holi_derot[:, 0]= (uvd_holi_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_holi_derot[:, 1]=(uvd_holi_derot[:, 1] - vmin )/urange*hand_wid+12

        uvd_hier_recur[:, 0] = (uvd_hier_recur[:, 0] - umin )/urange*hand_wid+12
        uvd_hier_recur[:, 1]= (uvd_hier_recur[:, 1] - vmin )/urange*hand_wid+12

        uvd_hybrid[:, 0]= (uvd_hybrid[:, 0] - umin )/urange*hand_wid+12
        uvd_hybrid[:, 1]=(uvd_hybrid[:, 1] - vmin )/urange*hand_wid+12

        i_gs=0
        subplot_row(fig,r0[i,12:84,12:84],gs,i_gs,uvd_true,uvd_hol,uvd_hol_derot,uvd_holi_derot,uvd_hier_recur,uvd_hybrid)

        v0_off=0.82
        vstep=0.3
        hstep=0.6/3.0
        h0_off = hstep/2+0.25
        ax0.text(h0_off-hstep,v0_off,'Ours:',ha='left', va='center',size=25,color='k')
        ax0.text(h0_off,v0_off,'GroundTruth',ha='center', va='center',size=20,color='k')
        ax0.text(h0_off+hstep*1,v0_off,'Hier_SA',ha='center', va='center',size=20,color='k')
        ax0.text(h0_off+hstep*2,v0_off,'Hybrid_Hier_SA',ha='center', va='center',size=20,color='k')

        v0_off=0.42
        ax0.text(h0_off-hstep,v0_off,'Baselines:',ha='left', va='center',size=25,color='k')
        ax0.text(h0_off,v0_off,'Holi',ha='center', va='center',size=20,color='k')
        ax0.text(h0_off+hstep*1,v0_off,'Holi_Derot',ha='center', va='center',size=20,color='k')
        ax0.text(h0_off+hstep*2,v0_off,'Holi_SA',ha='center', va='center',size=20,color='k')

        v0_off=0.47
        ax0.text(h0_off-hstep,v0_off,'Errors: ',ha='left', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_hier_recur[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        ax0.text(h0_off+hstep*1,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_hybrid[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        ax0.text(h0_off+hstep*2,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')

        v0_off=0.08
        ax0.text(h0_off-hstep,v0_off,'Errors: ',ha='left', va='center',size=20,color='k')
        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_pred_hol[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        ax0.text(h0_off,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_pred_hol_derot[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        ax0.text(h0_off+hstep*1,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_pred_holi_derot[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        ax0.text(h0_off+hstep*2,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')
        # plt.show()

        plt.savefig('C:/Proj/Proj_CNN_Hier/data/HDJIF_cmp_prior/%s_SELF_VIDEO_ECCV2016/%sself%04d.png'%(setname,setname,image_index+1),format='png',dpi=300)
        plt.close('all')

        # plt.savefig('C:/users/QiYE/OneDrive/Doc_ProgressReport/iros2016/%s/%d.png'%(setname,image_index),format='png', dpi=1200,bbox_inches='tight',frameon=False,face_color='k',pad_inches=0.1)