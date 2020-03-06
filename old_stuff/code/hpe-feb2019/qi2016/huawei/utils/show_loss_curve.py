__author__ = 'QiYE'

import numpy

import matplotlib.pyplot as plt

save_dir = 'F:/HuaweiProj/HuaWei_Seconddelivery_20180122/data/mega'
def show_loss_curve_singleloss2():
    version='pixel_aug_ker32_lr0.001000'
    loss = numpy.load('%s/detector_aug/loss_history_%s.npy'%(save_dir,version))
    train_loss = loss[0][1:]
    test_loss=loss[1][1:]
    x_axis=train_loss.shape[0]
    # x_axis=25
    loc = numpy.argmin(test_loss)
    print(version,'min test cost',test_loss[loc],train_loss[loc])

    plt.figure()
    plt.xlim(xmin=0,xmax=x_axis)
    plt.plot(numpy.arange(0,x_axis,1),train_loss[0:x_axis,], 'blue')
    plt.plot(numpy.arange(0,x_axis,1),test_loss[0:x_axis,],  c='r')
    # plt.yscale('log')ocmin]]*x_axis, '--', c='r')
    # p
    plt.grid('on','minor')
    plt.tick_params(which='minor' )
    plt.title(version)
    plt.show()



if __name__ == '__main__':
    show_loss_curve_singleloss2()

