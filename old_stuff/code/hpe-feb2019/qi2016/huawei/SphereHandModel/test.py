__author__ = 'QiYE'

# import random
# a = [[4,56],[7,8],[3,7],[7,8]]
# c = random.sample(a,2)
# print c
#
# c = random.sample(4,2)
# print c

from scipy.spatial.distance import cdist
import numpy


a = numpy.array([[4,56],[7,8],[3,7],[7,8]])

idx1=numpy.array([0,3])
idx2=numpy.array([1,0])
print (numpy.max([a[:,0],a[:,1]],axis=0))
print (a[(idx1,idx2)])



# print a1.shape
# a2 = numpy.array(a[0:2])
# print cdist(a1,a2).shape

# import matlab.engine
#
# eng = matlab.engine.start_matlab()
# # eng.edit('triarea',nargout=0)
#
# # eng.triarea(nargout=0)
# ret = eng.triarea(1.0,5.0)
# print(ret)

# import numpy as np
#
# a0 = np.array([[ 12.66427144 , 12.3593558 ,  12.41891346], [ 11.43537323 , 11.27939065 , 11.43138638]])
# b = np.array([[ 37.99281432 , 36.79569936,  36.8824373 ], [ 32.83431858 , 32.04327436 , 32.3835374 ]])
# c = np.array([[ 50.65708576 , 48.96058292 , 49.04048455], [ 43.27345305 , 42.10659575,  42.53240882]])
#
# a = np.array([[25.32854288,  24.60073771,  24.67798227], [ 22.23581958 , 21.78849162 , 22.05079657]])
# print 20.6309075471 - 10.6420468459
# print a-a0
#
# print  30.3948913868 - 20.6309075471
# print b-a
# print 40.0040263439 -30.3948913868
# print c-b

