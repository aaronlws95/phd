__author__ = 'QiYE'

import h5py
import numpy
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import constants, xyz_result_path
from src.utils.xyz_uvd import  xyz2uvd,uvd2xyz


def convert_uvd_to_xyz(uvd):
    res_x = 640
    res_y = 480

    scalefactor = 1
    focal_length_x = 0.8925925 * scalefactor
    focal_length_y =1.190123339 * scalefactor
    # focal_length = numpy.sqrt(focal_length_x ^ 2 + focal_length_y ^ 2);
    xyz = numpy.empty_like(uvd)
    z =  uvd[:,:,2]/1000 # convert mm to m
    xyz[:,:,2]=z
    xyz[:,:,0] = ( uvd[:,:,0] - res_x / 2)/res_x/ focal_length_x*z
    xyz[:,:,1] = ( -uvd[:,:,1] + res_y / 2)/res_y/focal_length_y*z

    return xyz

def convert_xyz_to_uvd(xyz):


    res_x = 640
    res_y = 480

    scalefactor = 1
    focal_length_x = 0.8925925 * scalefactor
    focal_length_y =1.190123339 * scalefactor

    uvd = numpy.empty_like(xyz)

    trans_x= xyz[:,:,0]
    trans_y= xyz[:,:,1]
    trans_z = xyz[:,:,2]
    uvd[:,:,0] = res_x / 2 + res_x * focal_length_x * ( trans_x / trans_z )
    uvd[:,:,1] = res_y / 2 - res_y * focal_length_y * ( trans_y / trans_z )
    uvd[:,:,2] = trans_z*1000 #convert m to mm

    return uvd


def subplot_row(fig,depth,gs,i_gs,xyz_true,xyz_pred_hol,xyz_pred_hol_derot,xyz_pred_holi_derot):
    linewidth=2
    markersize=7
    ax_wid = 71
    ax = fig.add_subplot(gs[i_gs])
    ax.imshow(depth,'gist_yarg')
    dot = xyz_true

    for k in [1,2,3,4,5]:

        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]

        ax.plot(x,y,linewidth=linewidth,marker='o',markersize=markersize,c='m')

        x=[dot[k,0],dot[k+5,0]]
        y=[dot[k,1],dot[k+5,1]]

        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)

    ax.set_xlim(0, ax_wid)
    ax.set_ylim(ax_wid,0 )
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+1])
    ax.imshow(depth,'gist_yarg')
    dot = xyz_pred_hol

    for k in [1,2,3,4,5]:

        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]

        ax.plot(x,y,linewidth=linewidth,marker='o',markersize=markersize,c='m')

        x=[dot[k,0],dot[k+5,0]]
        y=[dot[k,1],dot[k+5,1]]

        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)

    ax.set_xlim(0, ax_wid)
    ax.set_ylim(ax_wid,0 )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+2])
    ax.imshow(depth,'gist_yarg')
    dot = xyz_pred_hol_derot
    for k in [1,2,3,4,5]:

        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]

        ax.plot(x,y,linewidth=linewidth,marker='o',markersize=markersize,c='m')

        x=[dot[k,0],dot[k+5,0]]
        y=[dot[k,1],dot[k+5,1]]

        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)
    ax.set_xlim(0, ax_wid)
    ax.set_ylim(ax_wid,0 )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+3])
    ax.imshow(depth,'gist_yarg')
    dot = xyz_pred_holi_derot

    for k in [1,2,3,4,5]:

        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]

        ax.plot(x,y,linewidth=linewidth,marker='o',markersize=markersize,c='m')

        x=[dot[k,0],dot[k+5,0]]
        y=[dot[k,1],dot[k+5,1]]

        ax.plot(x,y,c=jnt_colors[k],linewidth=linewidth,marker='o',markersize=markersize)

    ax.set_xlim(0, ax_wid)
    ax.set_ylim(ax_wid,0 )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()



if __name__ =='__main__':

    uvd_fb= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\ICCV15_Feedback\\ICCV15_NYU_Feedback.txt')
    uvd_fb.shape=(uvd_fb.shape[0],14,3)
    uvd_hd= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\ICCV15_Feedback\\CVWW15_NYU_Prior-Refinement.txt')
    uvd_hd.shape=(uvd_hd.shape[0],14,3)
    part_jnt_idx_obw = [12,10,7,5,3,1,8,6,4,2,0]


    dataset_path_prefix=constants.Data_Path
    setname='NYU'
    dataset ='test'

    source_name= xyz_result_path.nyu_source_name_raw



    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']

    dataset_dir =  xyz_result_path.msrc_raw_dataset_dir
    part_jnt_idx = [0,2,6,10,14,18,4,8,12,16,20]
    xyz_pred_holi_derot_all=numpy.empty_like(xyz_true)
    for i in xrange(21):
        file=scipy.io.loadmat('D:\\nyu_tmp\\jnt%d_xyz.mat'%i)
        xyz_pred_holi_derot_all[:,i,:]=file['jnt']

    xyz_pred_holi_derot = xyz_pred_holi_derot_all[:,part_jnt_idx,:]
    xyz_true_part= xyz_true[:,part_jnt_idx,:]



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


    xyz_fb_all = uvd2xyz(setname,uvd_fb,roixy)
    xyz_hd_all = uvd2xyz(setname,uvd_hd,roixy)
    xyz_hd=xyz_hd_all[:,part_jnt_idx_obw,:]
    xyz_fb=xyz_fb_all[:,part_jnt_idx_obw,:]
    # uvd_fb=convert_xyz_to_uvd(xyz_fb_all)
    # uvd_hd=convert_xyz_to_uvd(xyz_hd_all)

    for image_index in xrange(0,500,5):
        print 'image_index',image_index

        fig = plt.figure(figsize=(16, 12))
        ax0 = fig.add_axes([0., 0., 1., 1., ], axisbg='k')
        ax0.text(0.5,0.85,'%s Test Sequence '%setname,ha='center', va='bottom',size=25,color='w')
        gs = gridspec.GridSpec(1,4,left=0.05, right=0.95,bottom=0.4,top=0.7,wspace=0.0, hspace=0)
        h0_off = 0.16
        v0_off=0.75
        plt.text(h0_off,v0_off,'GroundTruth',ha='center', va='center',size=20,color='w')
        plt.text(h0_off+0.225,v0_off,'HandsDeep',ha='center', va='center',size=20,color='w')
        plt.text(h0_off+0.225*2,v0_off,'FeedLoop',ha='center', va='center',size=20,color='w')
        plt.text(h0_off+0.225*3,v0_off,'Ours',ha='center', va='center',size=20,color='w')

        v0_off=0.35

        plt.text(h0_off,v0_off,'Errors: ',ha='center', va='center',size=20,color='w')
        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_hd[image_index]-xyz_true_part[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+0.225,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='w')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_fb[image_index]-xyz_true_part[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+0.225*2,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='w')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_pred_holi_derot[image_index]-xyz_true_part[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+0.225*3,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='w')


        vmin=rect_d1d2w[image_index,0]
        umin =rect_d1d2w[image_index,1]
        urange=rect_d1d2w[image_index,2]

        uvd_true = xyz2uvd(setname,xyz_true_part[image_index],roixy,jnt_type='single')
        uvd_holi_derot= xyz2uvd(setname,xyz_pred_holi_derot[image_index],roixy,jnt_type='single')


        uvd_handdeep= uvd_hd[image_index][part_jnt_idx_obw]
        uvd_feedback= uvd_fb[image_index][part_jnt_idx_obw]
        hand_wid = 72
        uvd_true[:, 0] = (uvd_true[:, 0] - umin )/urange*hand_wid+12
        uvd_true[:, 1]=(uvd_true[:, 1] - vmin )/urange*hand_wid+12
        uvd_handdeep[:, 0]= (uvd_handdeep[:, 0] - umin )/urange*hand_wid+12
        uvd_handdeep[:, 1]=(uvd_handdeep[:, 1] - vmin )/urange*hand_wid+12
        uvd_feedback[:, 0]= (uvd_feedback[:, 0] - umin )/urange*hand_wid+12
        uvd_feedback[:, 1]=(uvd_feedback[:, 1] - vmin )/urange*hand_wid+12
        uvd_holi_derot[:, 0]= (uvd_holi_derot[:, 0] - umin )/urange*hand_wid+12
        uvd_holi_derot[:, 1]=(uvd_holi_derot[:, 1] - vmin )/urange*hand_wid+12
        i_gs=0


        v_off = 0.15
        h_off = 0.015

        h0_off = 0.15
        v0_off = 0.03
        subplot_row(fig,r0[image_index,12:84,12:84],gs,i_gs,uvd_true,uvd_handdeep,uvd_feedback,uvd_holi_derot)
        ax0.text(0.9,0.25,'Frame Number: %d'%(image_index+1),ha='right', va='center',size=20,color='w')


        # plt.show()
        plt.savefig('C:/users/QiYE/OneDrive/Doc_ProgressReport/Hier_Hand_Error_Analysis_Figures/nyu_cmp_sequence/nyu_cmp_%03d.png'%(image_index+1),format='png',dpi=300)
        plt.close('all')

        # plt.savefig('C:/users/QiYE/OneDrive/Doc_ProgressReport/iros2016/%s/%d.png'%(setname,image_index),format='png', dpi=1200,bbox_inches='tight',frameon=False,face_color='k',pad_inches=0.1)