__author__ = 'QiYE'

import numpy
import scipy.io
from src.utils import constants
import matplotlib.pyplot as plt
import csv

dataset_path_prefix=constants.Data_Path
setname='icvl'
dataset='test'

keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
xyz_true = keypoints['xyz']



path = 'D:/Project/3DHandPose/Data_3DHandPoseDataset/Exp_Result_Prior_Work/Tang/eccv15/Final_Results/icvl/Best_Results_ICVL_N=25_M=30_20151030.csv'
xyz=  numpy.empty((1596,66),dtype='float32')
with open(path, 'rb') as f:

    reader = csv.reader(f.read().splitlines())
    i=0
    for row in reader:
        xyz[i]=row[1:67]
        i+=1

f.close()
idx_21 =numpy.array([1,2,3,4,17,5,6,7,18,8,9,10,19,11,12,13,20,14,15,16,21])-1

setname = 'icvl'
dataset='test'

xyz.shape = (xyz.shape[0],22,3)
xyz_hso_21jnt = xyz[:,idx_21,:]

idx = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
tmp = numpy.sqrt(numpy.sum((xyz_hso_21jnt - xyz_true)**2,axis=-1))*1000
err_hso = numpy.max(tmp[:,idx],axis=-1)
print '21 joint hso error ',numpy.mean(numpy.mean(tmp))
print '15 joint hso error ',numpy.mean(numpy.mean(tmp[:,idx]))



xyz_hybrid =scipy.io.loadmat('%sdata/HDJIF_cmp_prior/ICVL_hybrid_v2/ICVL_21_16.mat'% (dataset_path_prefix))['CNNResultDist2GT']
err_hybrid_v2 = numpy.max(xyz_hybrid[:,idx],axis=-1)
print '21 joint hybrid_v2 error ',numpy.mean(numpy.mean(xyz_hybrid))
print '15 joint hybrid_v2 error ',numpy.mean(numpy.mean(xyz_hybrid[:,idx]))




# xyz_hybrid =scipy.io.loadmat('%sdata/HDJIF_cmp_prior/ICVL_hybrid_v2/ICVL_21_16.mat'% (dataset_path_prefix))['CNNResultDist2GT']
# err_hybrid_v2 = numpy.max(xyz_hybrid,axis=-1)
# print '21 joint hybrid_v2 error ',numpy.mean(numpy.mean(xyz_hybrid))
# tmp = numpy.sqrt(numpy.sum((xyz_hso_21jnt - xyz_true)**2,axis=-1))*1000
# err_hso = numpy.max(tmp,axis=-1)
# print '21 joint hso error ',numpy.mean(numpy.mean(tmp))

maxd=80
thresholds=range(0,maxd,1)
all_size = err_hso.shape[0]
accuracy_hso = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hso <= thresholds[t])
    accuracy_hso[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

accuracy_hybrid  = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hybrid_v2  <= thresholds[t])
    accuracy_hybrid[t] = 100.0 * loc_bl_t[0].shape[0] / all_size


plt.figure()
c1, = plt.plot(thresholds,accuracy_hybrid[0:maxd],linewidth=1,c='r', hold=True)

c2, = plt.plot(thresholds, accuracy_hso[0:maxd],linewidth=1,c='b', hold=True)

plt.legend([c1,c2], ['Ours','HSO'],loc=4)
plt.xlabel('error threshold D (mm)',fontsize=20)
plt.ylabel('proportion of frames with all joint errors < D',fontsize=15)
plt.yticks(range(0,101,10))

plt.grid('on')
plt.show()
# plt.savefig('C:/Users/QiYE/OneDrive/eccv2016v3/images/icvlcmp.eps',format='eps', dpi=300)
# plt.savefig('C:/Users/QiYE/OneDrive/Doc_ProgressReport/eccv2016/images/icvlcmp.eps',format='eps', dpi=300)


