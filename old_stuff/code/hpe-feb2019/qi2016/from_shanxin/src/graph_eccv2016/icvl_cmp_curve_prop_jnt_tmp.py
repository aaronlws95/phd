__author__ = 'QiYE'

import numpy
import scipy.io
from src.utils import constants
import matplotlib.pyplot as plt
import csv

dataset_path_prefix=constants.Data_Path
setname='icvl'
dataset='test'

keypoints = scipy.io.loadmat('D:\Prj_CNN_Hier_v2\data\ICVL\source/test_icvl_xyz_21joints.mat')
xyz_true = keypoints['xyz']

xyz_hybrid =scipy.io.loadmat('D://Prj_CNN_Hier_v2//data//final_result/ICVL_hybrid_v2/ICVL_21_16.mat')
err_hybrid=xyz_hybrid.reshape((xyz_hybrid.shape[0]*xyz_hybrid.shape[1],1))
print '21jnts hybrid_v2 error ',numpy.mean(numpy.mean(xyz_hybrid))

path = 'D:\Prj_CNN_Hier_v2\data\Prior_Work_Result\Tang\Final_Results\Final_Results\ICVL\Best_Results_ICVL_N=25_M=30_20151030.csv'
xyz=  numpy.empty((1596,66),dtype='float32')
with open(path, 'rb') as f:

    reader = csv.reader(f.read().splitlines())
    i=0
    for row in reader:
        xyz[i]=row[1:67]
        i+=1

f.close()
idx =numpy.array([1,2,3,4,17,5,6,7,18,8,9,10,19,11,12,13,20,14,15,16,21])-1

setname = 'icvl'
dataset='test'

xyz.shape = (xyz.shape[0],22,3)
xyz_hso_21jnt = xyz[:,idx,:]

print 'hso error ',numpy.mean(numpy.mean( numpy.sqrt(numpy.sum((xyz_hso_21jnt - xyz_true)**2,axis=-1))))
tmp = numpy.sqrt(numpy.sum((xyz_hso_21jnt - xyz_true)**2,axis=-1))*1000
err_hso =tmp.reshape((tmp.shape[0]*tmp.shape[1],1))

maxd=30
thresholds=range(0,maxd,1)
all_size = err_hso.shape[0]
accuracy_hso = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hso <= thresholds[t])
    accuracy_hso[t] = 100.0 * loc_bl_t[0].shape[0] / all_size
accuracy_hybrid  = numpy.zeros((len(thresholds)),dtype='float32')
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hybrid  <= thresholds[t])
    accuracy_hybrid[t] = 100.0 * loc_bl_t[0].shape[0] / all_size


plt.figure()
c1, = plt.plot(thresholds,accuracy_hybrid[0:maxd],linewidth=1,c='r', hold=True)

c2, = plt.plot(thresholds, accuracy_hso[0:maxd],linewidth=1,c='b', hold=True)

plt.legend([c1,c2], ['Ours','HSO[19]'],loc=4)
plt.xlabel('error threshold D (mm)',fontsize=20)
plt.ylabel('proportion of joints within error < D',fontsize=18)
plt.yticks(range(0,101,10))

plt.grid('on')
# plt.show()
# plt.savefig('C:/Users/QiYE/OneDrive/eccv2016v3/images/icvlcmp.eps',format='eps', dpi=300)
plt.savefig('C:/Proj/Proj_Hier_Hand_eccv2016_07042016/Proj_CNN_Hier/data/HDJIF_cmp_prior/eccv2016_fig/icvlhybridv2joint.eps',format='eps', dpi=300)


