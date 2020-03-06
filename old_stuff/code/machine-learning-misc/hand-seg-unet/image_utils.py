import cv2
from matplotlib import pyplot as plt
import numpy as np

def preprocess(img, dim, channels):
    img = cv2.resize(img, dim)
    img = np.reshape(img, dim[0]*dim[1]*channels)
    img = img.astype(np.float32)
    img = (img - np.mean(img))/np.std(img)
    # img = (img - np.amin(img))/(np.amin(img) - np.amax(img))
    return img

def preprocess_target(img, dim, channels):
    img = img.astype(np.float32)
    img = cv2.resize(img, dim)
    img = np.reshape(img, (dim[0]*dim[1]*channels,1))
    img = img.astype(np.bool)
    return img

def restore(img, dim, channels):
    if channels == 1:
        img = np.reshape(img, (dim[0], dim[1]))
    else:
        img = np.reshape(img, (dim[0], dim[1], channels))
    return img

def show(img, channels, cmap=None):
    if channels == 3:
        # RGB --> BGR
        plt.imshow(img[..., ::-1], cmap=cmap)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()

def show_predict_and_target(predict, target):
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(predict[..., ::-1])
    plt.title('predict')

    plt.subplot(1, 2, 2)
    plt.imshow(target[..., ::-1])
    plt.title('target')

    plt.show()


def info(img, name='UNKNOWN'):
    print('NAME:', name)
    print('SHAPE:', img.shape)
    print('TYPE:', img.dtype)

