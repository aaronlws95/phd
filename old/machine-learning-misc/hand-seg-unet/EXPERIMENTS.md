# Exp 1

Baseline unet

input size: 256x256

data split ratio (train:test:val): 36:8:4

# Exp 2

Pre-trained VGG16 encoder unet

input size: 224x224 (augmented)

data split ratio (train:test:val): 36:8:4


# Notes

Exp 1 has better training accuracy and loss values. However, testing on real unseen images with `real_time_process.py` shows the segmentation has a clear bias towards colour (it would segment any body part or skin coloured objects). Could experiment with training the u-net on greyscale images.
