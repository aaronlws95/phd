__author__ = 'QiYE'
import numpy
import scipy.io
import theano

from src.utils import constants, xyz_result_path


dataset_path_prefix=constants.Data_Path
setname='msrc'
dataset ='test'



if setname =='icvl':
    hol_path = xyz_result_path.icvl_hol_path
    hol_derot_path = xyz_result_path.icvl_hol_derot_path
    hier_xyz_result= xyz_result_path.icvl_hier_xyz_result
    holi_xyz_result= xyz_result_path.icvl_holi_recur_xyz_result
    max_error =81

if setname =='nyu':
    hol_path = xyz_result_path.nyu_hol_path
    hol_derot_path = xyz_result_path.nyu_hol_derot_path
    hier_xyz_result = xyz_result_path.nyu_hier_xyz_result
    holi_xyz_result = xyz_result_path.nyu_holi_recur_xyz_result
    max_error = 40

if setname =='msrc':
    hol_path = xyz_result_path.msrc_hol_path
    hol_derot_path = xyz_result_path.msrc_hol_derot_path
    hier_xyz_result= xyz_result_path.msrc_hier_xyz_result
    holi_xyz_result= xyz_result_path.msrc_holi_recur_xyz_result
    max_error = 81

thresholds=range(0,max_error,1)


keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
xyz_true = keypoints['xyz']

# print xyz_true.shape
xyz_pred_hol= numpy.load('%sdata/%s/whole/best/%s.npy' % (dataset_path_prefix,setname,hol_path))[0]
print xyz_pred_hol.shape

xyz_pred_hol_derot= numpy.load('%sdata/%s/whole_derot/best/%s.npy' % (dataset_path_prefix,setname,hol_derot_path))[0]
print xyz_pred_hol_derot.shape

# mid2 = numpy.load('%sdata/%s/hier_derot_recur/final_xyz_uvd/%s.npy' % (dataset_path_prefix,setname,hier_xyz_result[6]))[0]
# mid2_true= xyz_true[:,2,:]
# err =  numpy.mean(numpy.sqrt(numpy.sum((mid2 - mid2_true)**2,axis=-1)),axis=0)*1000
# print 'mid err', err

idx =[0,
      1,6,11,16,
      2,7,12,17,
      3,8,13,18,
      4,9,14,19,
      5,10,15,20]


xyz_pred_hier_derot=numpy.empty_like(xyz_true)
xyz_pred_holi_derot=numpy.empty_like(xyz_true)

print xyz_pred_hol_derot.shape
for i,i_idx in enumerate(idx):
    path='%sdata/%s/hier_derot_recur/final_xyz_uvd/%s.npy' % (dataset_path_prefix,setname,hier_xyz_result[i_idx])
    xyz_pred_hier_derot[:,i,:] =numpy.load(path)[0]

    path='%sdata/%s/holi_derot_recur/final_xyz_uvd/%s.npy' % (dataset_path_prefix,setname,holi_xyz_result[i])
    xyz_pred_holi_derot[:,i,:]=numpy.load(path)[0]


err_hol = numpy.max(numpy.sqrt(numpy.sum((xyz_pred_hol - xyz_true)**2,axis=-1)),axis=-1)*1000
# print 'err_hol', numpy.mean(err_hol)
err_hol_derot = numpy.max(numpy.sqrt(numpy.sum((xyz_pred_hol_derot - xyz_true)**2,axis=-1)),axis=-1)*1000
# print 'err_hol_derot', numpy.mean(err_hol_derot)
err_hier_DJIF = numpy.max(numpy.sqrt(numpy.sum((xyz_pred_hier_derot - xyz_true)**2,axis=-1)),axis=-1)*1000
# print 'err_hier_derot', numpy.mean(err_hier_DJIF)
err_holi_DJIF = numpy.max(numpy.sqrt(numpy.sum((xyz_pred_holi_derot - xyz_true)**2,axis=-1)),axis=-1)*1000
# print 'err_holi_derot', numpy.mean(err_holi_DJIF)

all_size = xyz_true.shape[0]


accuracy_hol = numpy.zeros((len(thresholds)),dtype=theano.config.floatX)
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hol <= thresholds[t])
    accuracy_hol[t] = 100.0 * loc_bl_t[0].shape[0] / all_size


accuracy_hol_derot = numpy.zeros((len(thresholds)),dtype=theano.config.floatX)
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hol_derot <= thresholds[t])
    accuracy_hol_derot[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

accuracy_hier_DJIF = numpy.zeros((len(thresholds)),dtype=theano.config.floatX)
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_hier_DJIF <= thresholds[t])
    accuracy_hier_DJIF[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

accuracy_holi_DJIF = numpy.zeros((len(thresholds)),dtype=theano.config.floatX)
for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err_holi_DJIF <= thresholds[t])
    accuracy_holi_DJIF[t] = 100.0 * loc_bl_t[0].shape[0] / all_size

# scipy.io.savemat('%sdata/HDJIF_cmp_prior/%s_holi_percent_%d_21jnts' % (dataset_path_prefix,setname,max_error),{'hDJIF':accuracy_hol})
# scipy.io.savemat('%sdata/HDJIF_cmp_prior/%s_holi_derot_percent_%d_21jnts' % (dataset_path_prefix,setname,max_error),{'hDJIF':accuracy_hol})
scipy.io.savemat('%sdata/HDJIF_cmp_prior/%s_holi_DJIF_percent_%d_21jnts' % (dataset_path_prefix,setname,max_error),{'hDJIF':accuracy_hol})
scipy.io.savemat('%sdata/HDJIF_cmp_prior/%s_hDJIF_percent_%d_21jnts' % (dataset_path_prefix,setname,max_error),{'hDJIF':accuracy_holi_DJIF})



# plt.figure()
# c1, = plt.plot(thresholds, accuracy_hier_DJIF,'r',linewidth=1,hold=True)
# c2, = plt.plot(thresholds, accuracy_holi_DJIF,'y', linewidth=1,hold=True)
# c3, = plt.plot(thresholds,accuracy_hol_derot,'g',linewidth=1,hold=True )
# c4, = plt.plot(thresholds,accuracy_hol ,'b', linewidth=1)
#
# plt.legend([c1,c2,c3,c4], ['HDJIF', 'Holi_DJIF','Holi_Derot','Holi'],loc=4)
# plt.xlabel('error threshold D (mm)',fontsize=20)
# plt.ylabel('% frames with error < D',fontsize=20)
# plt.yticks(range(0,101,10))
#
# plt.grid('on')
# # plt.show()
#
# plt.savefig('%sdata/HDJIF_cmp_prior/%scurve' % (dataset_path_prefix,setname))

#
# plt.figure()
# c1, = plt.plot(thresholds, accuracy_holi_DJIF,'r',linewidth=1,hold=True)
# c2, = plt.plot(thresholds,accuracy_hol_derot,'g',linewidth=1,hold=True )
# c3, = plt.plot(thresholds,accuracy_hol ,'b', linewidth=1)
#
# plt.legend([c1,c2,c3], ['DJIF','Holi_Derot','Holi'],loc=4)
# plt.xlabel('error threshold D (mm)',fontsize=20)
# plt.ylabel('% frames with error < D',fontsize=20)
# plt.yticks(range(0,101,10))
# plt.grid('on')
# # plt.show()
# plt.savefig('C:/users/QiYE/OneDrive/Doc_ProgressReport/iros2016/ieeeconf/images/%scurve.eps' % setname,format='eps', dpi=300)
#




