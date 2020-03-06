# sphinx_gallery_thumbnail_number = 3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy
def show_two_imgs(backimg,topimg,alpha):

    # First we'll plot these blobs using only ``imshow``.
    vmax = topimg.max()
    vmin =  topimg.min()
    cmap = plt.cm.jet

    # Create an alpha channel of linearly increasing values moving to the right.
    alphas = np.ones(topimg.shape)*alpha
    # alphas[:, 30:] = np.linspace(1, 0, 70)

    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
    colors = Normalize(vmin, vmax, clip=True)(topimg)
    colors = cmap(colors)

    # Now set the alpha channel to the one we created above
    colors[..., -1] = alphas
    return backimg,colors,cmap

    # Create the figure and image
    # Note that the absolute values may be slightly different

    # plt.show()

import cv2
if __name__=='__main__':
    backimg=numpy.zeros((100,100))
    backimg[30:80,30:80]=1
    topimg = numpy.arange(0,10000,1).reshape(100,100)


    imgcopy=backimg.copy()

    min = imgcopy.min()
    max = imgcopy.max()
    imgcopy = (imgcopy - min) / (max - min) * 255.
    imgcopy = imgcopy.astype('uint8')
    imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)

    loc_hand = numpy.where(topimg>5000)
    mask_RGB = numpy.zeros_like(imgcopy)
    mask_RGB[loc_hand]=[255,0,0]

    alpha=0.1
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(imgcopy, alpha, mask_RGB, beta, 0.0)


    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', dst)
    cv2.waitKey()


    #
    # # topimg=numpy.zeros((100,100))
    # # topimg[10:80,50:80]=1
    # backimg,colors,cmap = show_two_imgs(backimg,topimg,0.1)
    # fig, ax = plt.subplots()
    # ax.imshow(backimg,'gray')
    # ax.imshow(colors,cmap=cmap)
    # plt.show()
