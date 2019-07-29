__author__ = 'QiYE'

__author__ = 'QiYE'

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from src.SphereHandModel.utils import xyz_uvd
from src.SphereHandModel.CostFunction import cost_function
from src.SphereHandModel.ShowSamples import ShowPointCloudFromDepth2
import scipy.io
import numpy
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
THUMB=[1,2,3,4]
INDEX=[5,6,7,8]
MIDDLE=[9,10,11,12]
RING=[13,14,15,16]
PINKY=[17,18,19,20]
WRIST=0
PALM=[1,5,9,13,17]

base_dir='F:/pami2017/Proj_CNN_Hier_v2_sx_icvl/'

M=45
xyz_true =scipy.io.loadmat('%sdata/icvl/source/test_icvl_xyz_21joints.mat'%base_dir)['xyz']*1000
xyz_cnn = numpy.empty_like(xyz_true)
for i in range(21):
    xyz_cnn[:,i,:]=scipy.io.loadmat('F:/pami2017/Prj_CNN_Hier_v2/data/icvl/ICVL_hier_derot_recur_v2/jnt%d_xyz.mat'%(i))['jnt']*1000
mean_err= numpy.mean((numpy.sqrt(numpy.sum((xyz_cnn- xyz_true)**2,axis=-1))))
print('hiercnn',mean_err)
cost_matrix=numpy.load('%sdata/done_v1/icvl_M500_N1000_Rang10/cost_pso_M500_N1000_range10.npy'%(base_dir))[:,0:100]
minloc=numpy.argmin(cost_matrix,axis=1)
# print('cost',cost_list)
xyz_pred = scipy.io.loadmat('%sdata/done_v1/icvl_M500_N1000_Rang10/pso_N99_M500_range10_seg0.mat'%(base_dir))['jnt']*1000
xyz_min=xyz_pred[(range(cost_matrix.shape[0]),minloc)]

err_pso=numpy.sqrt(numpy.sum((xyz_min-xyz_true)**2,axis=-1))
err_jnt_pso=err_pso.flatten()
err_cnn =numpy.sqrt(numpy.sum((xyz_cnn-xyz_true)**2,axis=-1))
err_jnt_cnn=err_cnn.flatten()

loc = numpy.where(err_jnt_cnn<30)
print(loc[0].shape[0]*1.0/err_jnt_cnn.shape[0])
print(numpy.mean(err_jnt_cnn[loc]))
print(numpy.mean(err_jnt_pso[loc]))



