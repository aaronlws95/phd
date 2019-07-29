__author__ = 'QiYE'
import numpy
import scipy.io
import matplotlib.pyplot as plt

from src.utils import constants

def convert_xyz_to_uvd(xyz):
    halfResX = 640/2
    halfResY = 480/2
    coeffX = 588.036865
    coeffY = 587.075073
    uvd = numpy.zeros(xyz.shape)
    uvd[:, :, 0] = numpy.asarray(coeffX * xyz[:, :, 0] / xyz[:, :, 2] +halfResX, dtype='float')
    uvd[:, :, 1] = numpy.asarray(halfResY+coeffY * xyz[:, :, 1] / xyz[:, :, 2], dtype='float')
    uvd[:, :, 2] = xyz[:, :, 2]*1000
    return uvd



dataset_path_prefix=constants.Data_Path
setname='nyu'
dataset='test'


thresholds=range(0,40,1)

part_jnt_idx_for_us = [0,2,6,10,14,18,4,8,12,16,20]
keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints_ori.mat' % (dataset_path_prefix,setname,dataset,setname))
uvd_true = convert_xyz_to_uvd(keypoints['xyz'][:,:,:])

xyz_hybrid = numpy.empty((uvd_true.shape[0],21,3))
for i in xrange(0,21,1):
    # holi_derot_recur_xyz[:,i,:]=scipy.io.loadmat('%sdata/HDJIF_cmp_prior/ICVL_hier_derot_recur_v2_1050/jnt%d_xyz.mat'% (dataset_path_prefix,i))['jnt']
    xyz_hybrid[:,i,:]=scipy.io.loadmat('%sdata/HDJIF_cmp_prior/NYU_hybrid_v2/jnt%d_xyz.mat'% (dataset_path_prefix,i))['jnt']
uvd_hybrid=convert_xyz_to_uvd(xyz_hybrid[:,:,:])

tmp = numpy.sqrt(numpy.sum((uvd_hybrid[:,part_jnt_idx_for_us,0:2] - uvd_true[:,part_jnt_idx_for_us,0:2])**2,axis=-1))
print 'uv error hybrid', numpy.mean(numpy.mean(tmp))
err_hybrid =tmp.reshape((tmp.shape[0]*tmp.shape[1],1))





uvd_nyu = scipy.io.loadmat('D:/Project/3DHandPose/Data_3DHandPoseDataset/NYU_dataset/NYU_dataset/test/test_predictions.mat')['pred_joint_uvconf'][0,:,:,0:2]
uvd_nyu_gt = scipy.io.loadmat('D:/Project/3DHandPose/Data_3DHandPoseDataset/NYU_dataset/NYU_dataset/test/joint_data.mat')['joint_uvd'][0,:,:,0:2]

part_jnt_idx_for_nyu =numpy.array([32,28,22,16,10,4,25,19,13,7,1])-1
tmp = numpy.sqrt(numpy.sum((uvd_nyu[:,part_jnt_idx_for_nyu,0:2] - uvd_nyu_gt[:,part_jnt_idx_for_nyu,0:2])**2,axis=-1))
print 'uv error nyu', numpy.mean(numpy.mean(tmp))
err_nyu =tmp.reshape((tmp.shape[0]*tmp.shape[1],1))

xyz_hso = scipy.io.loadmat('C:/Proj/Proj_Hier_Hand_eccv2016_07042016/Proj_CNN_Hier/data/HDJIF_cmp_prior/Result_Prior_Cmp/%s_hso_8252_21jnts'%setname)['xyz']
uvd_hso = convert_xyz_to_uvd(xyz_hso[:,part_jnt_idx_for_us,:])

tmp = numpy.sqrt(numpy.sum((uvd_hso[:,:,0:2] - uvd_true[:,:,0:2])**2,axis=-1))
print 'uv error hso', numpy.mean(numpy.mean(tmp))
err_hso =tmp.reshape((tmp.shape[0]*tmp.shape[1],1))



part_jnt_idx_for_obw = [12,10,7,5,3,1,8,6,4,2,0]
uvd_handdeep= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\Oberweger\\CVWW15_NYU_Prior-Refinement.txt')
uvd_handdeep.shape=(uvd_handdeep.shape[0],14,3)

tmp = numpy.sqrt(numpy.sum((uvd_handdeep[:,part_jnt_idx_for_obw,0:2] - uvd_nyu_gt[:,part_jnt_idx_for_nyu,0:2])**2,axis=-1))
print 'uv error uvd_handdeep', numpy.mean(numpy.mean(tmp))
err_hd =tmp.reshape((tmp.shape[0]*tmp.shape[1],1))


uvd_feedback= numpy.loadtxt('D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\Exp_Result_Prior_Work\\Oberweger\\ICCV15_NYU_Feedback.txt')
uvd_feedback.shape=(uvd_feedback.shape[0],14,3)

tmp = numpy.sqrt(numpy.sum((uvd_feedback[:,part_jnt_idx_for_obw,0:2] - uvd_nyu_gt[:,part_jnt_idx_for_nyu,0:2])**2,axis=-1))
print 'uv error uvd_feedback', numpy.mean(numpy.mean(tmp))
err_fb =tmp.reshape((tmp.shape[0]*tmp.shape[1],1))


all_size = err_hso.shape[0]
accuracy_hso = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hso <= thresholds[t])
    accuracy_hso[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

accuracy_hd = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hd <= thresholds[t])
    accuracy_hd[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

accuracy_fb = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_fb <= thresholds[t])
    accuracy_fb[t] = 100.0 * loc_bl_t[0].shape[0] / all_size


accuracy_hybrid  = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hybrid  <= thresholds[t])
    accuracy_hybrid[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

accuracy_nyu = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_nyu <= thresholds[t])
    accuracy_nyu[t] = 100.0 * loc_bl_t[0].shape[0] / all_size


plt.figure()

c2, = plt.plot(thresholds,accuracy_fb,'g',linewidth=1, hold=True )

c3, = plt.plot(thresholds, accuracy_hd,'c',linewidth=1, hold=True)

c4,=  plt.plot(thresholds, accuracy_hso,'b' ,linewidth=1,hold=True)

c5,=  plt.plot(thresholds, accuracy_nyu,'y' ,linewidth=1,hold=True)
c1, = plt.plot(thresholds, accuracy_hybrid,'r' ,linewidth=1)
plt.legend([c1,c4,c2,c3,c5], ['Ours','HSO','FeedLoop', 'HandsDeep','Tompson et al'],loc=4)

plt.xlabel('UV error threshold D (pixel)',fontsize=20)
plt.ylabel('proportion of joints within error < D',fontsize=18)
plt.yticks(range(0,101,10))

plt.grid('on')
plt.show()


# plt.savefig('C:/Users/QiYE/OneDrive/eccv2016v3/images/nyucmpuv.eps',format='eps', dpi=300)
# plt.savefig('C:/Users/QiYE/OneDrive/Doc_ProgressReport/eccv2016 submission/images/nyucmpuv.eps',format='eps', dpi=300)

