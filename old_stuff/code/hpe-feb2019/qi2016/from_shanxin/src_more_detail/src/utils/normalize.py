__author__ = 'QiYE'
import numpy

def norm_01(x):
    num=x.shape[0]
    chan = x.shape[1]
    img_size = x.shape[2]

    x.shape=(num*chan,img_size*img_size)
    min_v = numpy.min(x,axis=-1)
    max_v = numpy.max(x,axis=-1)
    range_v = max_v - min_v
    loc = numpy.where(range_v == 0)
    range_v[loc]=1.0
    min_v[loc]=0
    x=(x-min_v[:,numpy.newaxis])/range_v[:,numpy.newaxis]

    x.shape=(num,chan,img_size,img_size)
    return x
