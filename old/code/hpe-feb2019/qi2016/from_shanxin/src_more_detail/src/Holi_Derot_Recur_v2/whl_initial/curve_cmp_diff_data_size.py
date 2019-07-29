__author__ = 'icvl'

import numpy
import matplotlib.pyplot as plt
setname='mega'
dataset_path_prefix='/home/icvl/Qi/Prj_CNN_Hier_NewData/'
model_name= 'uvd_whl_r012_21jnts_half_64_96_128_1_2_adam_lm300_best'
path = '%sdata/%s/whole/best/param_cost_%s.npy'%(dataset_path_prefix,setname,model_name)

model_info = numpy.load(path)
train_cost = numpy.array(model_info[-2][1:-1])
test_cost = numpy.array(model_info[-1][1:-1])
print len(test_cost)
#for i in xrange(len(test_cost)):
#	print test_cost[i]

model_name= 'uvd_whl_r012_21jnts_quater_64_96_128_1_2_adam_lm300_best'
path = '%sdata/%s/whole/best/param_cost_%s.npy'%(dataset_path_prefix,setname,model_name)

model_info = numpy.load(path)
train_cost_quater = numpy.array(model_info[-2][1:-1])
test_cost_quater = numpy.array(model_info[-1][1:-1])
print len(test_cost_quater)
#for i in xrange(len(test_cost_quater)):
#	print test_cost_quater[i]

model_name= 'uvd_whl_r012_21jnts_8_64_96_128_1_2_adam_lm300_best'
path = '%sdata/%s/whole/best/param_cost_%s.npy'%(dataset_path_prefix,setname,model_name)

model_info = numpy.load(path)
train_cost_8 = numpy.array(model_info[-2][1:-1])
test_cost_8 = numpy.array(model_info[-1][1:-1])
print len(test_cost_8)
#for i in xrange(len(test_cost_quater)):
#	print test_cost_quater[i]
model_name= 'uvd_whl_r012_21jnts_16_64_96_128_1_2_adam_lm300_best'
path = '%sdata/%s/whole/best/param_cost_%s.npy'%(dataset_path_prefix,setname,model_name)

model_info = numpy.load(path)
train_cost_16 = numpy.array(model_info[-2][1:-1])
test_cost_16 = numpy.array(model_info[-1][1:-1])
print len(test_cost_16)
#for i in xrange(len(test_cost_quater)):
#	print test_cost_quater[i]

fig = plt.figure()

plt.xlim(xmin=1,xmax=400)
x_axis = len(test_cost)-1
plt.plot(range(1,x_axis*8,8),train_cost[0:x_axis,], 'r--')
plt.plot(numpy.arange(1,x_axis*8,8),test_cost[0:x_axis,], 'r')
x_axis = len(test_cost_quater)-1
plt.plot(numpy.arange(1,x_axis*4,4),train_cost_quater[0:x_axis,], 'b--')
plt.plot(numpy.arange(1,x_axis*4,4),test_cost_quater[0:x_axis,], 'b')

x_axis = len(test_cost_8)-1
plt.plot(numpy.arange(1,x_axis,1),train_cost_8[0:x_axis-1,], 'y--')
plt.plot(numpy.arange(1,x_axis,1),test_cost_8[0:x_axis-1,], 'y')
# plt.plot(numpy.arange(1,x_axis*2,2),train_cost_8[0:x_axis,], 'y--')
# plt.plot(numpy.arange(1,x_axis*2,2),test_cost_8[0:x_axis,], 'y')
x_axis = len(test_cost_16)-1
plt.plot(numpy.arange(1,x_axis,1),train_cost_16[0:x_axis-1,], 'm--')
plt.plot(numpy.arange(1,x_axis,1),test_cost_16[0:x_axis-1,], 'm')
plt.yscale('log')
plt.grid('on','minor')
plt.tick_params(which='minor' )
# plt.savefig('curve_cmp_diff_size.png')
plt.show()
