__author__ = 'QiYE'
import h5py
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy
def load_data_multi(batch_size,path,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')

    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    uvd = f['joint_label_uvd'][...][:,jnt_idx,:]

    f.close()

    # for i in numpy.random.randint(0,r0.shape[0],5):
    #     fig = plt.figure()
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(uvd[i,:,0]*72+12,uvd[i,:,1]*72+12,c='y',s=50)
    #     plt.show()
    if is_shuffle:
        r0,r1,r2,uvd = shuffle(r0,r1,r2,uvd,random_state=0)

        num = batch_size - r0.shape[0]%batch_size


        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])
    else:
        return r0.reshape(r0.shape[0], 1, r0.shape[1],r0.shape[2]),r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),\
               r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),uvd.reshape(uvd.shape[0], uvd.shape[1]*uvd.shape[2])