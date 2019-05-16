__author__ = 'QiYE'
import numpy
import scipy.io
import matplotlib.pyplot as plt

from src.utils import constants

def convert_uvd_to_xyz(uvd):
    xRes = 640
    yRes = 480
    xzfactor = 1.08836701
    yzfactor = 0.817612648
    normalizedX = numpy.asarray(uvd[:, :, 0], dtype='float32') / xRes - 0.5
    normalizedY = -numpy.asarray(uvd[:, :, 1], dtype='float32') / yRes + 0.5
    xyz = numpy.zeros(uvd.shape)
    xyz[:, :, 2] = numpy.asarray(uvd[:, :, 2], dtype='float32')
    xyz[:, :, 0] = normalizedX * xyz[:, :, 2] * xzfactor
    xyz[:, :, 1] = normalizedY * xyz[:, :, 2] * yzfactor
    return xyz
dataset_path_prefix=constants.Data_Path
setname='nyu'
dataset='test'


thresholds=range(0,81,1)


keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname))
xyz_true = keypoints['xyz']

print xyz_true.shape


part_jnt_idx_for_us = [0,2,6,10,14,18,4,8,12,16,20]
holi_derot_recur_xyz = numpy.empty((xyz_true.shape[0],21,3))
for i in xrange(0,21,1):
    # holi_derot_recur_xyz[:,i,:]=scipy.io.loadmat('%sdata/HDJIF_cmp_prior/ICVL_hier_derot_recur_v2_1050/jnt%d_xyz.mat'% (dataset_path_prefix,i))['jnt']
    holi_derot_recur_xyz[:,i,:]=scipy.io.loadmat('%sdata/HDJIF_cmp_prior/NYU_holi_derot_recur_v2/jnt%d_xyz.mat'% (dataset_path_prefix,i))['jnt']


part_jnt_idx_for_obw = [12,10,7,5,3,1,8,6,4,2,0]

uvd_handdeep= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\Oberweger\\CVWW15_NYU_Prior-Refinement.txt')
uvd_handdeep.shape=(uvd_handdeep.shape[0],14,3)


uvd_feedback= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\Oberweger\\ICCV15_NYU_Feedback.txt')
uvd_feedback.shape=(uvd_feedback.shape[0],14,3)
xyz_hd = convert_uvd_to_xyz(uvd_handdeep)
xyz_fb= convert_uvd_to_xyz(uvd_feedback)


nyu_xyz_true = scipy.io.loadmat('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\NYU_dataset\\NYU_dataset\\testTable\\joint_data.mat')['joint_xyz'][0]
print nyu_xyz_true.shape
part_jnt_idx_for_nyu_true=[30,27,21,15,9,3,24,18,12,6,0]
print '21jnts holi_derot_recur_v2 error ',numpy.mean(numpy.mean( numpy.sqrt(numpy.sum((holi_derot_recur_xyz - xyz_true)**2,axis=-1))))
print '11jnts holi_derot_recur_v2 error ',numpy.mean(numpy.mean( numpy.sqrt(numpy.sum((holi_derot_recur_xyz[:,part_jnt_idx_for_us,:] - xyz_true[:,part_jnt_idx_for_us,:])**2,axis=-1))))



err_holi_derot_recur_v2 = numpy.max(numpy.sqrt(numpy.sum((holi_derot_recur_xyz[:,part_jnt_idx_for_us,:] - xyz_true[:,part_jnt_idx_for_us,:])**2,axis=-1))*1000,axis=-1)



print '11jnts handsdeep error ',numpy.mean(numpy.mean( numpy.sqrt(numpy.sum((xyz_hd[:,part_jnt_idx_for_obw,:] - nyu_xyz_true[:,part_jnt_idx_for_nyu_true,:])**2,axis=-1))))

err_hd = numpy.max(numpy.sqrt(numpy.sum((xyz_hd[:,part_jnt_idx_for_obw,:] - nyu_xyz_true[:,part_jnt_idx_for_nyu_true,:])**2,axis=-1)),axis=-1)
print '11jnts feedback error ',numpy.mean(numpy.mean( numpy.sqrt(numpy.sum((xyz_fb[:,part_jnt_idx_for_obw,:] - nyu_xyz_true[:,part_jnt_idx_for_nyu_true,:])**2,axis=-1))))


jnt14_idx = [0,3,6,9,12,15,18,20,24,25,27,30,31,32]

print '14jnts handsdeep error ',numpy.mean(numpy.mean( numpy.sqrt(numpy.sum((xyz_hd- nyu_xyz_true[:,jnt14_idx,:])**2,axis=-1))))
print '14jnts feedback error ',numpy.mean(numpy.mean( numpy.sqrt(numpy.sum((xyz_fb - nyu_xyz_true[:,jnt14_idx,:])**2,axis=-1))))

err_fb = numpy.max(numpy.sqrt(numpy.sum((xyz_fb[:,part_jnt_idx_for_obw,:] - nyu_xyz_true[:,part_jnt_idx_for_nyu_true,:])**2,axis=-1)),axis=-1)



all_size = err_fb.shape[0]


accuracy_hd = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hd <= thresholds[t])
    accuracy_hd[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

accuracy_fb = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_fb <= thresholds[t])
    accuracy_fb[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

accuracy_holi  = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_holi_derot_recur_v2  <= thresholds[t])
    accuracy_holi[t] = 100.0 * loc_bl_t[0].shape[0] / all_size


plt.figure()
c1, = plt.plot(thresholds, accuracy_holi,'r' ,linewidth=1,hold=True)
c2, = plt.plot(thresholds,accuracy_fb,'g',linewidth=1, hold=True )

c3, = plt.plot(thresholds, accuracy_hd,'c',linewidth=1, hold=True)



plt.legend([c1,c2,c3], ['Ours','FeedLoop', 'HandsDeep',],loc=4)

plt.xlabel('error threshold D (mm)',fontsize=20)
plt.ylabel('proportion of frames with all joint errors < D (%)',fontsize=15)
plt.ylabel('% frames with error < D',fontsize=20)
# plt.ylabel('proportion of frames with all joint errors < D (%)',fontsize=15)
plt.yticks(range(0,101,10))

plt.grid('on')
plt.show()
# plt.savefig('C:/Proj/Proj_Hier_Hand_eccv2016_07042016/Proj_CNN_Hier/data/HDJIF_cmp_prior/iros2016_fig/nyucmpmax.eps',format='eps', dpi=300)
# plt.savefig('C:/Users/QiYE/OneDrive/Doc_ProgressReport/eccv2016/images/nyucmp.eps',format='eps', dpi=300)

