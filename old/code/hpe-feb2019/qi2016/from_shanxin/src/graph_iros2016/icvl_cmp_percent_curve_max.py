__author__ = 'QiYE'

import numpy
import scipy.io
from src.utils import constants
import matplotlib.pyplot as plt
import theano
dataset_path_prefix=constants.Data_Path
setname='icvl'


joint_num = 21,

color = [(255/255.0, 56/255.0, 37/255.0),
         (0/255.0, 118/255.0, 255/255.0),
         (84/255.0 ,200/255.0, 253/255.0),
         (67/255.0 ,219/255.0, 94/255.0),
         (255/255.0 ,206/255.0, 0/255.0),
         (250/255.0, 98/255.0, 195/255.0),
         (255/255.0 ,150/255.0 ,0)]

# color = color - 0.2,
# color(color<0) = 0,

# YNeg_Hand_suc = [0,0.0144110275689223,0.125313283208020,0.241854636591479,0.409147869674186,0.576441102756892,0.722431077694236,0.824561403508772,0.897243107769424,0.933583959899749,0.954887218045113,0.969298245614035,0.982456140350877,0.993107769423559,0.996240601503759,0.998746867167920]
lrf_suc_rate =numpy.array([0,0.00626566416040100,0.0708020050125313,0.208020050125313,0.369674185463659,0.504385964912281,0.629699248120301,0.709273182957394,0.802005012531328,0.859022556390978,0.890977443609023,0.915413533834587,0.933583959899749,0.951754385964912,0.966791979949875,0.975563909774436])*100
# Full_Hand_Iter6_suc = [0,0.0501253132832080,0.241228070175439,0.446115288220551,0.617794486215539,0.748746867167920,0.827067669172932,0.884711779448622,0.924812030075188,0.944235588972431,0.961779448621554,0.974310776942356,0.981203007518797,0.985588972431078,0.991854636591479,0.997493734335840]
# Full_Hand_Iter1_suc = [0,0.00501253132832080,0.0488721804511278,0.231203007518797,0.474310776942356,0.655388471177945,0.757518796992481,0.844611528822055,0.891604010025063,0.921679197994987,0.947994987468672,0.964285714285714,0.976817042606516,0.984962406015038,0.991854636591479,0.995614035087719]
Sep_Hand_Iter9_suc = numpy.array([0,0.112155388471178,0.334586466165414,0.553884711779449,0.729949874686717,0.840852130325815,0.902255639097744,0.935463659147870,0.960526315789474,0.971177944862155,0.979949874686717,0.982456140350877,0.985588972431078,0.987468671679198,0.991228070175439,0.994360902255639])*100

Hso_suc_rate = numpy.array([0.000000,0.000000,0.052632,0.278195,0.545113,0.734962,0.834586,0.908521,0.941103,0.957393,0.971805,0.978697,0.986216,0.989975,0.992481,0.994361,0.995614,0.996241,0.996867,0.996867,0.997494])*100
Keskin_suc_rate =numpy.array([0,0,0,0,0.00050025,0.0025013,0.017009,0.048524,0.10205,0.15808,0.22511,0.30465,0.38919,0.46173,0.54227,0.61731,0.67684,0.73337,0.78789,0.82941,0.86843])*100
Melax_suc_rate = numpy.array( [0,0.041,0.057,0.081,0.121,0.144,0.171,0.192,0.208,0.247,0.27,0.306,0.355,0.417,0.471,0.518,0.568,0.603,0.648,0.69,0.735])*100

keypoints = scipy.io.loadmat('%sdata/%s/source/test_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,setname))
xyz_true = keypoints['xyz']
idx = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
holi_derot_recur_xyz = numpy.empty((xyz_true.shape[0],21,3))
for i in xrange(0,21,1):
    # holi_derot_recur_xyz[:,i,:]=scipy.io.loadmat('%sdata/HDJIF_cmp_prior/ICVL_hier_derot_recur_v2_1050/jnt%d_xyz.mat'% (dataset_path_prefix,i))['jnt']
    holi_derot_recur_xyz[:,i,:]=scipy.io.loadmat('%sdata/HDJIF_cmp_prior/ICVL_holi_derot_recur_v2/jnt%d_xyz.mat'% (dataset_path_prefix,i))['jnt']


thresholds=range(0,81,4)
holi_derot_recur_suc_rate = numpy.zeros((len(thresholds)),dtype=theano.config.floatX)
mean_err= numpy.mean((numpy.sqrt(numpy.sum((holi_derot_recur_xyz- xyz_true)**2,axis=-1))),axis=-1)*1000
print 'icvl holi_derot_recur_v2 error ', numpy.mean(mean_err)


# err= numpy.max(numpy.sqrt(numpy.sum((holi_derot_recur_xyz[:,1:21,:] - xyz_true[:,1:21,:])**2,axis=-1)),axis=-1)*1000
err= numpy.max((numpy.sqrt(numpy.sum((holi_derot_recur_xyz- xyz_true)**2,axis=-1)))[:,idx],axis=-1)*1000
# err= numpy.max((numpy.sqrt(numpy.sum((holi_derot_recur_xyz- xyz_true)**2,axis=-1))),axis=-1)*1000

for t in xrange(len(thresholds)):
    loc_bl_t = numpy.where(err <= thresholds[t])
    holi_derot_recur_suc_rate[t] = 100.0 * loc_bl_t[0].shape[0] / err.shape[0]


suc_rate_D = range(5,81,5)
plt.figure()
c5, = plt.plot(suc_rate_D, Sep_Hand_Iter9_suc,linewidth=1, c=color[4],hold=True)
c1, = plt.plot(thresholds,holi_derot_recur_suc_rate,linewidth=1,c=color[0], hold=True)

c2, = plt.plot(thresholds, Melax_suc_rate,linewidth=1,c=color[1], hold=True)
c3, = plt.plot(thresholds, Hso_suc_rate,linewidth=1,c=color[2], hold=True)
c4, = plt.plot(suc_rate_D, lrf_suc_rate,linewidth=1,c=color[3], hold=True)

c6, = plt.plot(thresholds, Keskin_suc_rate ,linewidth=1,c=color[5])
plt.legend([c1,c2,c3,c4,c5,c6], ['Ours','Melax et al.','HSO','LRF','Sun et al.','Keskin et al.'],loc=4)
plt.xlabel('error threshold D (mm)',fontsize=20)
plt.ylabel('% frames with error < D',fontsize=20)
# plt.ylabel('proportion of frames with all joint errors < D (%)',fontsize=15)
plt.yticks(range(0,101,10))

plt.grid('on')
# plt.show()
plt.savefig('C:/Proj/Proj_Hier_Hand_eccv2016_07042016/Proj_CNN_Hier/data/HDJIF_cmp_prior/iros2016_fig/icvlcmpmax.eps',format='eps', dpi=300)


