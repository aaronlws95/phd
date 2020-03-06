__author__ = 'QiYE'
import h5py
import numpy
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import constants, xyz_result_path
from src.utils.xyz_uvd import  xyz2uvd


def subplot_row(fig,depth,gs,i_gs,xyz_true,xyz_pred_hol,xyz_pred_hol_derot,xyz_pred_holi_derot):
    linewidth=2
    markersize=7
    ax_wid = 71
    ax = fig.add_subplot(gs[i_gs])
    ax.imshow(depth,'gist_yarg')
    dot = xyz_true

    for k in [1,5,9,13,17]:

        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]

        ax.plot(x,y,linewidth=linewidth,marker='o',markersize=markersize,c='m')

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]

        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)

    ax.set_xlim(0, ax_wid)
    ax.set_ylim(0, ax_wid)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+1])
    ax.imshow(depth,'gist_yarg')
    dot = xyz_pred_hol

    for k in [1,5,9,13,17]:

        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        ax.plot(x,y,c='m',linewidth=linewidth,marker='o',markersize=markersize)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)

    ax.set_xlim(0, ax_wid)
    ax.set_ylim(0, ax_wid)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+2])
    ax.imshow(depth,'gist_yarg')
    dot = xyz_pred_hol_derot
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        ax.plot(x,y,linewidth=linewidth,marker='o',c='m',markersize=markersize)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)

    ax.set_xlim(0, ax_wid)
    ax.set_ylim(0, ax_wid)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+3])
    ax.imshow(depth,'gist_yarg')
    dot = xyz_pred_holi_derot

    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        ax.plot(x,y,linewidth=linewidth,marker='o',c='m',markersize=markersize)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)
    ax.set_xlim(0, ax_wid)
    ax.set_ylim(0, ax_wid)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()


if __name__ =='__main__':


    dataset_path_prefix=constants.Data_Path
    setname='ICVL'
    dataset ='test'

    if setname =='ICVL':
        source_name= xyz_result_path.icvl_source_name_ori
        derot_whlimg= xyz_result_path.icvl_derot_whlimg
        hol_path = xyz_result_path.icvl_hol_path
        hol_derot_path = xyz_result_path.icvl_hol_derot_path
        hier_xyz_result= xyz_result_path.icvl_hier_xyz_result
        holi_xyz_result= xyz_result_path.icvl_holi_recur_xyz_result

    if setname =='NYU':
        source_name= xyz_result_path.nyu_source_name_ori
        derot_whlimg= xyz_result_path.nyu_derot_whlimg
        hol_path = xyz_result_path.nyu_hol_path
        hol_derot_path = xyz_result_path.nyu_hol_derot_path
        hier_xyz_result = xyz_result_path.nyu_hier_xyz_result
        holi_xyz_result = xyz_result_path.nyu_holi_recur_xyz_result

    if setname =='MSRC':
        source_name= xyz_result_path.msrc_source_name_ori

        derot_whlimg= xyz_result_path.msrc_derot_whlimg
        hol_path = xyz_result_path.msrc_hol_path
        hol_derot_path = xyz_result_path.msrc_hol_derot_path
        hier_xyz_result= xyz_result_path.msrc_hier_xyz_result
        holi_xyz_result= xyz_result_path.msrc_holi_recur_xyz_result

    xyz_pred_hol= numpy.load('%sdata/%s/whole/best/%s.npy' % (dataset_path_prefix,setname,hol_path))[0]
    print xyz_pred_hol.shape

    xyz_pred_hol_derot= numpy.load('%sdata/%s/whole_derot/best/%s.npy' % (dataset_path_prefix,setname,hol_derot_path))[0]
    print xyz_pred_hol_derot.shape

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']

    idx =[0,
          1,6,11,16,
          2,7,12,17,
          3,8,13,18,
          4,9,14,19,
          5,10,15,20]

    dataset_dir =  xyz_result_path.msrc_raw_dataset_dir
    kinect_index = 1


    xyz_pred_holi_derot=numpy.empty_like(xyz_true)
    for i,i_idx in enumerate(idx):
        path='%sdata/%s/holi_derot_recur/final_xyz_uvd/%s.npy' % (dataset_path_prefix,setname,holi_xyz_result[i])
        xyz_pred_holi_derot[:,i,:]=numpy.load(path)[0]

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    print xyz_true.shape
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_uvd_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    uvd_true = keypoints['uvd']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']

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

    uvd_true = xyz2uvd(setname,xyz_true,roixy,jnt_type=None)
    uvd_hol = xyz2uvd(setname,xyz_pred_hol,roixy,jnt_type=None)
    uvd_hol_derot = xyz2uvd(setname,xyz_pred_hol_derot,roixy,jnt_type=None)
    uvd_holi_derot= xyz2uvd(setname,xyz_pred_holi_derot,roixy,jnt_type=None)

    for image_index in rng1.randint(0,xyz_true.shape[0],20):
        print 'image_index',image_index
        # fig = plt.figure(figsize=(16, 12), facecolor='k')
        fig = plt.figure(figsize=(16, 12))
        ax0 = fig.add_axes([0., 0., 1., 1., ], axisbg='k')
        ax0.text(0.5,0.96,'%s Test Dataset'%setname,ha='center', va='bottom',size=25,color='w')
        gs = gridspec.GridSpec(3,4,left=0.05, right=0.95,bottom=0.0,top=0.90,wspace=0.0, hspace=0)
        h0_off = 0.16
        v0_off=0.93
        plt.text(h0_off,v0_off,'GroundTruth',ha='center', va='center',size=20,color='w')
        plt.text(h0_off+0.225,v0_off,'Holi',ha='center', va='center',size=20,color='w')
        plt.text(h0_off+0.225*2,v0_off,'Holi_Derot',ha='center', va='center',size=20,color='w')
        plt.text(h0_off+0.225*3,v0_off,'Ours',ha='center', va='center',size=20,color='w')

        v_off = 0.15
        h_off = 0.015

        h0_off = 0.15
        v0_off = 0.03

        i=image_index
        vmin=rect_d1d2w[i,0]
        umin =rect_d1d2w[i,1]
        urange=rect_d1d2w[i,2]

        uvd_true = xyz2uvd(setname,xyz_true[i],roixy,jnt_type='single')
        uvd_hol = xyz2uvd(setname,xyz_pred_hol[i],roixy,jnt_type='single')
        uvd_hol_derot = xyz2uvd(setname,xyz_pred_hol_derot[i],roixy,jnt_type='single')
        uvd_holi_derot= xyz2uvd(setname,xyz_pred_holi_derot[i],roixy,jnt_type='single')

        hand_wid = 72
        uvd_true[:, 0] = (uvd_true[:, 0] - umin )/urange*hand_wid+12
        uvd_true[:, 1]=(uvd_true[:, 1] - vmin )/urange*hand_wid+12
        uvd_hol[ :, 0] = (uvd_hol[ :, 0] - umin )/urange*hand_wid+12
        uvd_hol[:, 1]=(uvd_hol[:, 1] - vmin )/urange*hand_wid+12
        uvd_hol_derot[:, 0] = (uvd_hol_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_hol_derot[:, 1]= (uvd_hol_derot[:, 1] - vmin )/urange*hand_wid+12
        uvd_holi_derot[:, 0]= (uvd_holi_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_holi_derot[:, 1]=(uvd_holi_derot[:, 1] - vmin )/urange*hand_wid+12
        i_gs=0
        # subplot_row(fig,r0[i,12:84,12:84],gs,i_gs,uvd_true_i,uvd_hol_i,uvd_hol_derot_i,uvd_holi_derot_i)
        subplot_row(fig,r0[i,12:84,12:84],gs,i_gs,uvd_true,uvd_hol,uvd_hol_derot,uvd_holi_derot)
        ax0.text(h_off,v_off+0.3*2,'Frame Number: %d'%i,ha='left', va='center',size=15,color='w',rotation=90)

        i=image_index+1
        vmin=rect_d1d2w[i,0]
        umin =rect_d1d2w[i,1]
        urange=rect_d1d2w[i,2]

        uvd_true = xyz2uvd(setname,xyz_true[i],roixy,jnt_type='single')
        uvd_hol = xyz2uvd(setname,xyz_pred_hol[i],roixy,jnt_type='single')
        uvd_hol_derot = xyz2uvd(setname,xyz_pred_hol_derot[i],roixy,jnt_type='single')
        uvd_holi_derot= xyz2uvd(setname,xyz_pred_holi_derot[i],roixy,jnt_type='single')

        hand_wid = 72
        uvd_true[:, 0] = (uvd_true[:, 0] - umin )/urange*hand_wid+12
        uvd_true[:, 1]=(uvd_true[:, 1] - vmin )/urange*hand_wid+12
        uvd_hol[ :, 0] = (uvd_hol[ :, 0] - umin )/urange*hand_wid+12
        uvd_hol[:, 1]=(uvd_hol[:, 1] - vmin )/urange*hand_wid+12
        uvd_hol_derot[:, 0] = (uvd_hol_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_hol_derot[:, 1]= (uvd_hol_derot[:, 1] - vmin )/urange*hand_wid+12
        uvd_holi_derot[:, 0]= (uvd_holi_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_holi_derot[:, 1]=(uvd_holi_derot[:, 1] - vmin )/urange*hand_wid+12
        i_gs=4
        subplot_row(fig,r0[i,12:84,12:84],gs,i_gs,uvd_true,uvd_hol,uvd_hol_derot,uvd_holi_derot)
        ax0.text(h_off,v_off+0.3*1,'Frame Number: %d'%i,ha='left', va='center',size=15,color='w',rotation=90)


        i=image_index+2
        vmin=rect_d1d2w[i,0]
        umin =rect_d1d2w[i,1]
        urange=rect_d1d2w[i,2]

        uvd_true = xyz2uvd(setname,xyz_true[i],roixy,jnt_type='single')
        uvd_hol = xyz2uvd(setname,xyz_pred_hol[i],roixy,jnt_type='single')
        uvd_hol_derot = xyz2uvd(setname,xyz_pred_hol_derot[i],roixy,jnt_type='single')
        uvd_holi_derot= xyz2uvd(setname,xyz_pred_holi_derot[i],roixy,jnt_type='single')

        hand_wid = 72
        uvd_true[:, 0] = (uvd_true[:, 0] - umin )/urange*hand_wid+12
        uvd_true[:, 1]=(uvd_true[:, 1] - vmin )/urange*hand_wid+12
        uvd_hol[ :, 0] = (uvd_hol[ :, 0] - umin )/urange*hand_wid+12
        uvd_hol[:, 1]=(uvd_hol[:, 1] - vmin )/urange*hand_wid+12
        uvd_hol_derot[:, 0] = (uvd_hol_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_hol_derot[:, 1]= (uvd_hol_derot[:, 1] - vmin )/urange*hand_wid+12
        uvd_holi_derot[:, 0]= (uvd_holi_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_holi_derot[:, 1]=(uvd_holi_derot[:, 1] - vmin )/urange*hand_wid+12
        i_gs=8
        subplot_row(fig,r0[i,12:84,12:84],gs,i_gs,uvd_true,uvd_hol,uvd_hol_derot,uvd_holi_derot)
        ax0.text(h_off,v_off,'Frame Number: %d'%i,ha='left', va='center',size=15,color='w',rotation=90)


        # plt.show()
        plt.savefig('C:/users/QiYE/OneDrive/Doc_ProgressReport/iros2016/%s/%d.png'%(setname,image_index),format='png',dpi=300)
        # plt.savefig('C:/users/QiYE/OneDrive/Doc_ProgressReport/iros2016/%s/%d.png'%(setname,image_index),format='png', dpi=1200,bbox_inches='tight',frameon=False,face_color='k',pad_inches=0.1)