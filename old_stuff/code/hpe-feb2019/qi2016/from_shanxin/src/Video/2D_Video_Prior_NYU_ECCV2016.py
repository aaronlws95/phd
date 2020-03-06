__author__ = 'QiYE'
import h5py
import numpy
import scipy.io
from src.utils import constants
import matplotlib.pyplot as plt
from src.utils import xyz_result_path

import matplotlib.gridspec as gridspec
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
    xyz[:,:,1] = ( uvd[:,:,1] - res_y / 2)/res_y/focal_length_y*z

    return xyz


def subplot_row(fig,depth,gs,i_gs,xyz_true,uvd_hso,uvd_handdeep,uvd_feedback,uvd_hybrid):
    linewidth=2
    markersize=7
    ax_wid = 71
    ax = fig.add_subplot(gs[i_gs])
    ax.imshow(depth,'gray')
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
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

    ax = fig.add_subplot(gs[i_gs+1])
    ax.imshow(depth,'gray')
    dot = uvd_hso
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
    ax.imshow(depth,'gray')
    dot = uvd_handdeep
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
    ax.imshow(depth,'gray')
    dot = uvd_feedback
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

    ax = fig.add_subplot(gs[i_gs+4])
    ax.imshow(depth,'gray')
    dot = uvd_hybrid
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


    dataset_path_prefix=constants.Data_Path
    setname='NYU'
    dataset ='test'
    part_jnt_idx_obw = [12,10,7,5,3,1,8,6,4,2,0]

    part_jnt_idx_us = [0,2,6,10,14,18,4,8,12,16,20]
    source_name=xyz_result_path.nyu_source_name_ori


    xyz_hybrid =numpy.empty((2000,11,3),dtype='float32')
    for i,idx in enumerate(part_jnt_idx_us):
        xyz_hybrid[:,i,:] = scipy.io.loadmat('C:/Proj/Proj_CNN_Hier/data/nyu/nyu_cnn_pso/jnt%d_xyz.mat'%idx)['jnt']

    xyz_hso = scipy.io.loadmat('C:/Users/QiYE/OneDrive/Doc_ProgressReport/eccv2016/images/%s_hso_slice'%setname)['xyz'][:,part_jnt_idx_us,:]


    uvd_fb= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\ICCV15_Feedback\\ICCV15_NYU_Feedback.txt')
    uvd_fb.shape=(uvd_fb.shape[0],14,3)
    uvd_hd= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\ICCV15_Feedback\\CVWW15_NYU_Prior-Refinement.txt')
    uvd_hd.shape=(uvd_hd.shape[0],14,3)

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz'][:,part_jnt_idx_us,:]
    print xyz_true.shape
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_uvd_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    uvd_true = keypoints['uvd'][:,part_jnt_idx_us,:]
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']

    dataset_dir =  xyz_result_path.msrc_raw_dataset_dir
    kinect_index = 1


    src_path='%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    r0=f['r0'][...]
    print r0.shape
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    derot_uvd = f['joint_label_uvd'][...]
    ori_test_idx=f['ori_test_idx'][...]
    f.close()


    xyz_fb_all = convert_uvd_to_xyz(uvd_fb)
    xyz_hd_all = convert_uvd_to_xyz(uvd_hd)
    xyz_hd=xyz_hd_all[:,part_jnt_idx_obw,:]
    xyz_fb=xyz_fb_all[:,part_jnt_idx_obw,:]

    cmap = plt.cm.rainbow
    colors_map = cmap(numpy.arange(cmap.N))
    rng = numpy.random.RandomState(0)
    num = rng.randint(0,256,(21,))
    jnt_colors = colors_map[num]
    print jnt_colors.shape
    for image_index in xrange(100,150,3):
        print 'image_index',image_index
        fig = plt.figure(figsize=(16, 12), facecolor='w')
        # fig = plt.figure(figsize=(9, 4))
        ax0 = fig.add_axes([0., 0., 1., 1., ], axisbg='w')
        ax0.text(0.5,0.85,'%s Test Sequence '%setname,ha='center', va='bottom',size=30,color='k')
        ax0.text(0.9,0.25,'Frame Number: %d'%(image_index+1),ha='right', va='center',size=20,color='k')
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.set_axis_off()
        gs = gridspec.GridSpec(1,5,left=0.05, right=0.95,bottom=0.4,top=0.7,wspace=0.0, hspace=0)

        v0_off=0.75
        step=1/5.0
        h0_off = step/2
        plt.text(h0_off,v0_off,'GroundTruth',ha='center', va='center',size=20,color='k')
        plt.text(h0_off+step,v0_off,'HSO',ha='center', va='center',size=20,color='k')
        plt.text(h0_off+step*2,v0_off,'HandsDeep',ha='center', va='center',size=20,color='k')
        plt.text(h0_off+step*3,v0_off,'FeedLoop',ha='center', va='center',size=20,color='k')
        plt.text(h0_off+step*4,v0_off,'Ours',ha='center', va='center',size=20,color='k')


        v0_off=0.35
        plt.text(h0_off,v0_off,'Errors: ',ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_hso[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+step*1,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_hd[ori_test_idx[image_index]]-xyz_true[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+step*2,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_fb[ori_test_idx[image_index]]-xyz_true[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+step*3,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')

        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_hybrid[image_index]-xyz_true[image_index])**2,axis=-1)))*1000
        plt.text(h0_off+step*4,v0_off,'%.0f mm'%err,ha='center', va='center',size=20,color='k')




        i=image_index
        vmin=rect_d1d2w[i,0]
        umin =rect_d1d2w[i,1]
        urange=rect_d1d2w[i,2]

        uvd_true = xyz2uvd(setname,xyz_true[i],roixy,jnt_type='single')
        uvd_hso= xyz2uvd(setname,xyz_hso[i],roixy,jnt_type='single')
        uvd_hybrid= xyz2uvd(setname,xyz_hybrid[i],roixy,jnt_type='single')

        uvd_handdeep= uvd_hd[ori_test_idx[image_index]][part_jnt_idx_obw]
        uvd_feedback= uvd_fb[ori_test_idx[image_index]][part_jnt_idx_obw]

        hand_wid = 72
        uvd_true[:, 0] = (uvd_true[:, 0] - umin )/urange*hand_wid+12
        uvd_true[:, 1]=(uvd_true[:, 1] - vmin )/urange*hand_wid+12

        uvd_hso[:, 0] = (uvd_hso[:, 0] - umin )/urange*hand_wid+12
        uvd_hso[:, 1]= (uvd_hso[:, 1] - vmin )/urange*hand_wid+12



        uvd_handdeep[:, 0]= (uvd_handdeep[:, 0] - umin )/urange*hand_wid+12
        uvd_handdeep[:, 1]=(uvd_handdeep[:, 1] - vmin )/urange*hand_wid+12

        uvd_feedback[:, 0]= (uvd_feedback[:, 0] - umin )/urange*hand_wid+12
        uvd_feedback[:, 1]=(uvd_feedback[:, 1] - vmin )/urange*hand_wid+12
        # print uvd_feedback, uvd_true
        uvd_hybrid[:, 0]= (uvd_hybrid[:, 0] - umin )/urange*hand_wid+12
        uvd_hybrid[:, 1]=(uvd_hybrid[:, 1] - vmin )/urange*hand_wid+12

        i_gs=0
        subplot_row(fig,r0[i,12:84,12:84],gs,i_gs,uvd_true,uvd_hso,uvd_handdeep,uvd_feedback,uvd_hybrid)



        # plt.show()
        # plt.savefig('C:/Proj/Proj_CNN_Hier/data/HDJIF_cmp_prior/%s_ECCV2016_CMP_EXAMP/%s%04d.eps'%(setname,setname,image_index+1),format='eps',dpi=300)
        plt.savefig('C:/Proj/Proj_CNN_Hier/data/HDJIF_cmp_prior/%s_ECCV2016_CMP_EXAMP/%sCMP%04d.png'%(setname,setname,image_index+1),format='png',dpi=300)
        plt.close('all')

        # plt.savefig('C:/users/QiYE/OneDrive/Doc_ProgressReport/iros2016/%s/%d.png'%(setname,image_index),format='png', dpi=1200,bbox_inches='tight',frameon=False,face_color='k',pad_inches=0.1)