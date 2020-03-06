__author__ = 'QiYE'
import h5py
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy
import scipy

def load_data_multi_half_train(batch_size,path,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle

    f = h5py.File(path,'r')

    r0 = f['r0'][...]
    print 'original train samples',r0.shape[0]
    idx = range(0,r0.shape[0],2)
    r0 = r0[idx]
    r1 = f['r1'][...][idx]
    r2= f['r2'][...][idx]
    uvd= f['uvd_jnt_gt_norm'][...][idx]

    f.close()

    # for i in numpy.random.randint(0,r0.shape[0],5):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0[i],'gray')
    #     ax.scatter(uvd[i,:,0]*96,uvd[i,:,1]*96,c='y',s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1[i],'gray')
    #     ax.scatter(uvd[i,:,0]*48,uvd[i,:,1]*48,c='y',s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2[i],'gray')
    #     ax.scatter(uvd[i,:,0]*24,uvd[i,:,1]*24,c='y',s=5)
    #     plt.show()
    print 'half num sample',r0.shape[0]
    num = batch_size - r0.shape[0]%batch_size

    if is_shuffle:
        r0,r1,r2,uvd = shuffle(r0,r1,r2,uvd,random_state=0)

        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])
    else:
        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])


def load_data_multi_rotate_2(batch_size,path,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle

    f = h5py.File(path,'r')

    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    uvd = f['uvd_jnt_gt_norm'][...][:,jnt_idx,:]

    f.close()
    print r0.shape
    # rotate the images

    # r01 = numpy.rot90(r0,0);
    # r11 = numpy.rot90(r1,0);
    # r21 = numpy.rot90(r2,0);
    # uvd1 = numpy.rot90(uvd,0);

    theta1 = 0.5 * numpy.pi
    rotMatrix1 = numpy.array([[numpy.cos(theta1), -numpy.sin(theta1), 0],
                              [numpy.sin(theta1), numpy.cos(theta1),  0],
                              [0                , 0                ,  1]])

    theta2 = 1.0 * numpy.pi
    rotMatrix2 = numpy.array([[numpy.cos(theta2), -numpy.sin(theta2), 0],
                              [numpy.sin(theta2), numpy.cos(theta2), 0],
                              [0, 0, 1]])

    theta3 = 1.5 * numpy.pi
    rotMatrix3 = numpy.array([[numpy.cos(theta3), -numpy.sin(theta3), 0],
                              [numpy.sin(theta3), numpy.cos(theta3), 0],
                              [0, 0, 1]])

    print r0.shape
    TotalNum = r0.shape[0]
    centertemp = 0.5 * numpy.ones((21, 3))

    RR0 = []
    RR1 = []
    RR2 = []
    UVDUVD = []

    # for j in range(0, 1000, 1):
    #
    for j in range(0,TotalNum,1):
        if j%50000 ==0:
            print 'rotating image '
            print j


        r0temp = numpy.rot90(r0[j],1)
        r1temp = numpy.rot90(r1[j],1)
        r2temp = numpy.rot90(r2[j],1)
        uvdtemp = numpy.dot(uvd[j] - centertemp, rotMatrix1) + centertemp

        r0temp2 = numpy.rot90(r0temp, 1)
        r1temp2 = numpy.rot90(r1temp, 1)
        r2temp2 = numpy.rot90(r2temp, 1)
        uvdtemp2 = numpy.dot(uvd[j] - centertemp, rotMatrix2) + centertemp

        r0temp3 = numpy.rot90(r0temp2, 1)
        r1temp3 = numpy.rot90(r1temp2, 1)
        r2temp3 = numpy.rot90(r2temp2, 1)
        uvdtemp3 = numpy.dot(uvd[j] - centertemp, rotMatrix3) + centertemp

        # RR0.append(r0temp)
        RR0.append(r0temp2)
        # RR0.append(r0temp3)

        # RR1.append(r1temp)
        RR1.append(r1temp2)
        # RR1.append(r1temp3)

        # RR2.append(r2temp)
        RR2.append(r2temp2)
        # RR2.append(r2temp3)

        # UVDUVD.append(uvdtemp)
        UVDUVD.append(uvdtemp2)
        # UVDUVD.append(uvdtemp3)



        # if 0 == j:
        #     RR0 = r0temp
        #     RR0 = numpy.stack([RR0, r0temp2, r0temp3], axis=1)
        #     RR1 = r1temp
        #     RR1 = numpy.stack([RR1, r1temp2, r1temp3], axis=1)
        #     RR2 = r2temp
        #     RR2 = numpy.stack([RR2, r2temp2, r2temp3], axis=1)
        #     UVDUVD = uvdtemp
        #     UVDUVD = numpy.stack([UVDUVD, uvdtemp2, uvdtemp3], axis=1)
        #     # RR0 = r0temp
        #     # RR0 = numpy.concatenate([RR0, r0temp2, r0temp3], axis=2)
        #     # RR1 = r1temp
        #     # RR1 = numpy.concatenate([RR1, r1temp2, r1temp3], axis=2)
        #     # RR2 = r2temp
        #     # RR2 = numpy.concatenate([RR2, r2temp2, r2temp3], axis=2)
        #     # UVDUVD = uvdtemp
        #     # UVDUVD = numpy.concatenate([UVDUVD, uvdtemp2, uvdtemp3], axis=2)
        #
        # else:
        #     RR0 = numpy.concatenate([RR0, r0temp, r0temp2, r0temp3], axis=2)
        #     RR1 = numpy.concatenate([RR1, r1temp, r1temp2, r1temp3], axis=2)
        #     RR2 = numpy.concatenate([RR2, r2temp, r2temp2, r2temp3], axis=2)
        #     UVDUVD = numpy.concatenate([UVDUVD, uvdtemp, uvdtemp2, uvdtemp3], axis=2)


        # data_array = numpy.array(RR0)
        # print data_array.shape

    RR0_arr = numpy.array(RR0, dtype='float32')
    RR1_arr = numpy.array(RR1, dtype='float32')
    RR2_arr = numpy.array(RR2, dtype='float32')
    UVDUVD_arr = numpy.array(UVDUVD, dtype='float32')

    print RR0_arr.shape
    print RR1_arr.shape
    print RR2_arr.shape
    print UVDUVD_arr.shape






    # for j in range(0,TotalNum,1):
    #     # original figures
    #
    #
    #     r0temp = numpy.rot90(r0[j],1)
    #     r1temp = numpy.rot90(r1[j],1)
    #     r2temp = numpy.rot90(r2[j],1)
    #     uvdtemp = numpy.dot(uvd[j] - centertemp, rotMatrix1) + centertemp
    #
    #     r0temp2 = numpy.rot90(r0temp, 1)
    #     r1temp2 = numpy.rot90(r1temp, 1)
    #     r2temp2 = numpy.rot90(r2temp, 1)
    #     uvdtemp2 = numpy.dot(uvd[j] - centertemp, rotMatrix2) + centertemp
    #
    #     r0temp3 = numpy.rot90(r0temp2, 1)
    #     r1temp3 = numpy.rot90(r1temp2, 1)
    #     r2temp3 = numpy.rot90(r2temp2, 1)
    #     uvdtemp3 = numpy.dot(uvd[j] - centertemp, rotMatrix3) + centertemp
    #
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0temp, 'gray')
    #     ax.scatter(uvdtemp[:, 0] * 96, uvdtemp[:, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1temp, 'gray')
    #     ax.scatter(uvdtemp[:, 0] * 48, uvdtemp[:, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2temp, 'gray')
    #     ax.scatter(uvdtemp[:, 0] * 24, uvdtemp[:, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0temp2, 'gray')
    #     ax.scatter(uvdtemp2[:, 0] * 96, uvdtemp2[:, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1temp2, 'gray')
    #     ax.scatter(uvdtemp2[:, 0] * 48, uvdtemp2[:, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2temp2, 'gray')
    #     ax.scatter(uvdtemp2[:, 0] * 24, uvdtemp2[:, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0temp3, 'gray')
    #     ax.scatter(uvdtemp3[:, 0] * 96, uvdtemp3[:, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1temp3, 'gray')
    #     ax.scatter(uvdtemp3[:, 0] * 48, uvdtemp3[:, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2temp3, 'gray')
    #     ax.scatter(uvdtemp3[:, 0] * 24, uvdtemp3[:, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0[j], 'gray')
    #     ax.scatter(uvd[j, :, 0] * 96, uvd[j, :, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1[j], 'gray')
    #     ax.scatter(uvd[j, :, 0] * 48, uvd[j, :, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2[j], 'gray')
    #     ax.scatter(uvd[j, :, 0] * 24, uvd[j, :, 1] * 24, c='y', s=5)
    #     plt.show()



    # for i in range(0, 1000, 5):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(RR0_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 96, UVDUVD_arr[i, :, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(RR1_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 48, UVDUVD_arr[i, :, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(RR2_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 24, UVDUVD_arr[i, :, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     i = i + 1
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(RR0_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 96, UVDUVD_arr[i, :, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(RR1_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 48, UVDUVD_arr[i, :, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(RR2_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 24, UVDUVD_arr[i, :, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     i = i + 2
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(RR0_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 96, UVDUVD_arr[i, :, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(RR1_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 48, UVDUVD_arr[i, :, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(RR2_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 24, UVDUVD_arr[i, :, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #







    r0 = numpy.concatenate([RR0_arr, r0], axis=0)
    r1 = numpy.concatenate([RR1_arr, r1], axis=0)
    r2 = numpy.concatenate([RR2_arr, r2], axis=0)
    uvd = numpy.concatenate([UVDUVD_arr, uvd], axis=0)

    # r0.astype(float32)
    # r1.astype(float32)
    # r2.astype(float32)
    # uvd.astype(float32)

    print r0.shape

    # for i in range(100,r0.shape[0],5):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0[i],'gray')
    #     ax.scatter(uvd[i,:,0]*96,uvd[i,:,1]*96,c='y',s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1[i],'gray')
    #     ax.scatter(uvd[i,:,0]*48,uvd[i,:,1]*48,c='y',s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2[i],'gray')
    #     ax.scatter(uvd[i,:,0]*24,uvd[i,:,1]*24,c='y',s=5)
    #     plt.show()

    print 'original num sample',r0.shape[0]
    num = batch_size - r0.shape[0]%batch_size

    if is_shuffle:
        r0,r1,r2,uvd = shuffle(r0,r1,r2,uvd,random_state=0)

        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])
    else:
        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])



def load_data_multi_rotate_4(batch_size,path,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle

    f = h5py.File(path,'r')

    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    uvd = f['uvd_jnt_gt_norm'][...][:,jnt_idx,:]

    f.close()
    print r0.shape
    # rotate the images

    # r01 = numpy.rot90(r0,0);
    # r11 = numpy.rot90(r1,0);
    # r21 = numpy.rot90(r2,0);
    # uvd1 = numpy.rot90(uvd,0);

    theta1 = 0.5 * numpy.pi
    rotMatrix1 = numpy.array([[numpy.cos(theta1), -numpy.sin(theta1), 0],
                              [numpy.sin(theta1), numpy.cos(theta1),  0],
                              [0                , 0                ,  1]])

    theta2 = 1.0 * numpy.pi
    rotMatrix2 = numpy.array([[numpy.cos(theta2), -numpy.sin(theta2), 0],
                              [numpy.sin(theta2), numpy.cos(theta2), 0],
                              [0, 0, 1]])

    theta3 = 1.5 * numpy.pi
    rotMatrix3 = numpy.array([[numpy.cos(theta3), -numpy.sin(theta3), 0],
                              [numpy.sin(theta3), numpy.cos(theta3), 0],
                              [0, 0, 1]])


    print r0.shape
    TotalNum = r0.shape[0]
    centertemp = 0.5 * numpy.ones((21, 3))

    RR0 = []
    RR1 = []
    RR2 = []
    UVDUVD = []

    # for j in range(0, 1000, 1):
    #
    for j in range(0,TotalNum,1):
        if j%50000 ==0:
            print 'rotating image '
            print j


        r0temp = numpy.rot90(r0[j],1)
        r1temp = numpy.rot90(r1[j],1)
        r2temp = numpy.rot90(r2[j],1)
        uvdtemp = numpy.dot(uvd[j] - centertemp, rotMatrix1) + centertemp

        r0temp2 = numpy.rot90(r0temp, 1)
        r1temp2 = numpy.rot90(r1temp, 1)
        r2temp2 = numpy.rot90(r2temp, 1)
        uvdtemp2 = numpy.dot(uvd[j] - centertemp, rotMatrix2) + centertemp

        r0temp3 = numpy.rot90(r0temp2, 1)
        r1temp3 = numpy.rot90(r1temp2, 1)
        r2temp3 = numpy.rot90(r2temp2, 1)
        uvdtemp3 = numpy.dot(uvd[j] - centertemp, rotMatrix3) + centertemp

        RR0.append(r0temp)
        RR0.append(r0temp2)
        RR0.append(r0temp3)

        RR1.append(r1temp)
        RR1.append(r1temp2)
        RR1.append(r1temp3)

        RR2.append(r2temp)
        RR2.append(r2temp2)
        RR2.append(r2temp3)

        UVDUVD.append(uvdtemp)
        UVDUVD.append(uvdtemp2)
        UVDUVD.append(uvdtemp3)



        # if 0 == j:
        #     RR0 = r0temp
        #     RR0 = numpy.stack([RR0, r0temp2, r0temp3], axis=1)
        #     RR1 = r1temp
        #     RR1 = numpy.stack([RR1, r1temp2, r1temp3], axis=1)
        #     RR2 = r2temp
        #     RR2 = numpy.stack([RR2, r2temp2, r2temp3], axis=1)
        #     UVDUVD = uvdtemp
        #     UVDUVD = numpy.stack([UVDUVD, uvdtemp2, uvdtemp3], axis=1)
        #     # RR0 = r0temp
        #     # RR0 = numpy.concatenate([RR0, r0temp2, r0temp3], axis=2)
        #     # RR1 = r1temp
        #     # RR1 = numpy.concatenate([RR1, r1temp2, r1temp3], axis=2)
        #     # RR2 = r2temp
        #     # RR2 = numpy.concatenate([RR2, r2temp2, r2temp3], axis=2)
        #     # UVDUVD = uvdtemp
        #     # UVDUVD = numpy.concatenate([UVDUVD, uvdtemp2, uvdtemp3], axis=2)
        #
        # else:
        #     RR0 = numpy.concatenate([RR0, r0temp, r0temp2, r0temp3], axis=2)
        #     RR1 = numpy.concatenate([RR1, r1temp, r1temp2, r1temp3], axis=2)
        #     RR2 = numpy.concatenate([RR2, r2temp, r2temp2, r2temp3], axis=2)
        #     UVDUVD = numpy.concatenate([UVDUVD, uvdtemp, uvdtemp2, uvdtemp3], axis=2)


        # data_array = numpy.array(RR0)
        # print data_array.shape

    RR0_arr = numpy.array(RR0, dtype='float32')
    RR1_arr = numpy.array(RR1, dtype='float32')
    RR2_arr = numpy.array(RR2, dtype='float32')
    UVDUVD_arr = numpy.array(UVDUVD, dtype='float32')

    print RR0_arr.shape
    print RR1_arr.shape
    print RR2_arr.shape
    print UVDUVD_arr.shape






    # for j in range(0,TotalNum,1):
    #     # original figures
    #
    #
    #     r0temp = numpy.rot90(r0[j],1)
    #     r1temp = numpy.rot90(r1[j],1)
    #     r2temp = numpy.rot90(r2[j],1)
    #     uvdtemp = numpy.dot(uvd[j] - centertemp, rotMatrix1) + centertemp
    #
    #     r0temp2 = numpy.rot90(r0temp, 1)
    #     r1temp2 = numpy.rot90(r1temp, 1)
    #     r2temp2 = numpy.rot90(r2temp, 1)
    #     uvdtemp2 = numpy.dot(uvd[j] - centertemp, rotMatrix2) + centertemp
    #
    #     r0temp3 = numpy.rot90(r0temp2, 1)
    #     r1temp3 = numpy.rot90(r1temp2, 1)
    #     r2temp3 = numpy.rot90(r2temp2, 1)
    #     uvdtemp3 = numpy.dot(uvd[j] - centertemp, rotMatrix3) + centertemp
    #
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0temp, 'gray')
    #     ax.scatter(uvdtemp[:, 0] * 96, uvdtemp[:, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1temp, 'gray')
    #     ax.scatter(uvdtemp[:, 0] * 48, uvdtemp[:, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2temp, 'gray')
    #     ax.scatter(uvdtemp[:, 0] * 24, uvdtemp[:, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0temp2, 'gray')
    #     ax.scatter(uvdtemp2[:, 0] * 96, uvdtemp2[:, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1temp2, 'gray')
    #     ax.scatter(uvdtemp2[:, 0] * 48, uvdtemp2[:, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2temp2, 'gray')
    #     ax.scatter(uvdtemp2[:, 0] * 24, uvdtemp2[:, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0temp3, 'gray')
    #     ax.scatter(uvdtemp3[:, 0] * 96, uvdtemp3[:, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1temp3, 'gray')
    #     ax.scatter(uvdtemp3[:, 0] * 48, uvdtemp3[:, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2temp3, 'gray')
    #     ax.scatter(uvdtemp3[:, 0] * 24, uvdtemp3[:, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0[j], 'gray')
    #     ax.scatter(uvd[j, :, 0] * 96, uvd[j, :, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1[j], 'gray')
    #     ax.scatter(uvd[j, :, 0] * 48, uvd[j, :, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2[j], 'gray')
    #     ax.scatter(uvd[j, :, 0] * 24, uvd[j, :, 1] * 24, c='y', s=5)
    #     plt.show()



    # for i in range(0, 1000, 5):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(RR0_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 96, UVDUVD_arr[i, :, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(RR1_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 48, UVDUVD_arr[i, :, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(RR2_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 24, UVDUVD_arr[i, :, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     i = i + 1
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(RR0_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 96, UVDUVD_arr[i, :, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(RR1_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 48, UVDUVD_arr[i, :, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(RR2_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 24, UVDUVD_arr[i, :, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #     i = i + 2
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(RR0_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 96, UVDUVD_arr[i, :, 1] * 96, c='y', s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(RR1_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 48, UVDUVD_arr[i, :, 1] * 48, c='y', s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(RR2_arr[i], 'gray')
    #     ax.scatter(UVDUVD_arr[i, :, 0] * 24, UVDUVD_arr[i, :, 1] * 24, c='y', s=5)
    #     plt.show()
    #
    #







    r0 = numpy.concatenate([RR0_arr, r0], axis=0)
    r1 = numpy.concatenate([RR1_arr, r1], axis=0)
    r2 = numpy.concatenate([RR2_arr, r2], axis=0)
    uvd = numpy.concatenate([UVDUVD_arr, uvd], axis=0)

    # r0.astype(float32)
    # r1.astype(float32)
    # r2.astype(float32)
    # uvd.astype(float32)

    print r0.shape

    # for i in range(100,r0.shape[0],5):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0[i],'gray')
    #     ax.scatter(uvd[i,:,0]*96,uvd[i,:,1]*96,c='y',s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1[i],'gray')
    #     ax.scatter(uvd[i,:,0]*48,uvd[i,:,1]*48,c='y',s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2[i],'gray')
    #     ax.scatter(uvd[i,:,0]*24,uvd[i,:,1]*24,c='y',s=5)
    #     plt.show()

    print 'original num sample',r0.shape[0]
    num = batch_size - r0.shape[0]%batch_size

    if is_shuffle:
        r0,r1,r2,uvd = shuffle(r0,r1,r2,uvd,random_state=0)

        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])
    else:
        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])


def load_data_multi(batch_size,path,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle

    f = h5py.File(path,'r')

    r0 = f['r0'][...]

    r1 = f['r1'][...]
    r2= f['r2'][...]
    uvd = f['uvd_jnt_gt_norm'][...][:,jnt_idx,:]

    f.close()
    print r0.shape
    # for i in numpy.random.randint(0,r0.shape[0],5):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0[i],'gray')
    #     ax.scatter(uvd[i,:,0]*96,uvd[i,:,1]*96,c='y',s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1[i],'gray')
    #     ax.scatter(uvd[i,:,0]*48,uvd[i,:,1]*48,c='y',s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2[i],'gray')
    #     ax.scatter(uvd[i,:,0]*24,uvd[i,:,1]*24,c='y',s=5)
    #     plt.show()
    print 'original num sample',r0.shape[0]
    num = batch_size - r0.shape[0]%batch_size

    if is_shuffle:
        r0,r1,r2,uvd = shuffle(r0,r1,r2,uvd,random_state=0)

        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])
    else:
        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])



def load_data_multi_shanxin(batch_size,trainingdata_name, Training_Subjects_Names,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle

    r0 = []
    r1 = []
    r2 = []
    uvd = []
    for x in range(0, len(Training_Subjects_Names)):
        current_subject = Training_Subjects_Names[x]
        path = '%s%s.h5' % (trainingdata_name, current_subject)
        print path
        f = h5py.File(path,'r')
        r0temp = f['r0'][...]
        r1temp = f['r1'][...]
        r2temp = f['r2'][...]
        uvdtemp = f['uvd_jnt_gt_norm'][...][:,jnt_idx,:]

        if 0 == x:
            r0 = r0temp
            r1 = r1temp
            r2 = r2temp
            uvd = uvdtemp
        else:
            r0 = numpy.concatenate([r0, r0temp], axis=0)
            r1 = numpy.concatenate([r1, r1temp], axis=0)
            r2 = numpy.concatenate([r2, r2temp], axis=0)
            uvd = numpy.concatenate([uvd, uvdtemp], axis=0)
        print r0temp.shape, r0.shape
        f.close()



    print r0.shape
    # for i in numpy.random.randint(0,r0.shape[0],5):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(311)
    #     ax.imshow(r0[i],'gray')
    #     ax.scatter(uvd[i,:,0]*96,uvd[i,:,1]*96,c='y',s=5)
    #     ax = fig.add_subplot(312)
    #     ax.imshow(r1[i],'gray')
    #     ax.scatter(uvd[i,:,0]*48,uvd[i,:,1]*48,c='y',s=5)
    #     ax = fig.add_subplot(313)
    #     ax.imshow(r2[i],'gray')
    #     ax.scatter(uvd[i,:,0]*24,uvd[i,:,1]*24,c='y',s=5)
    #     plt.show()
    print 'original num sample',r0.shape[0]
    num = batch_size - r0.shape[0]%batch_size

    if is_shuffle:
        r0,r1,r2,uvd = shuffle(r0,r1,r2,uvd,random_state=0)

        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])
    else:
        return numpy.concatenate([r0,r0[0:num]],axis=0).reshape(r0.shape[0]+num, 1, r0.shape[1],r0.shape[2]), \
               numpy.concatenate([r1,r1[0:num]],axis=0).reshape(r1.shape[0]+num, 1, r1.shape[1],r1.shape[2]),\
               numpy.concatenate([r2,r2[0:num]],axis=0).reshape(r2.shape[0]+num, 1, r2.shape[1],r2.shape[2]),\
               numpy.concatenate([uvd,uvd[0:num]],axis=0).reshape(uvd.shape[0]+num, uvd.shape[1]*uvd.shape[2])
