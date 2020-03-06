# Setup guide

Ubuntu 16.04

## Nvidia Driver

```
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update`
sudo apt-get install nvidia-410
```

## CUDA

go to Nvidia website and download CUDA 9.0 runfile(local)

`sudo sh <cuda-runfile>.run --silent --toolkit --toolkitpath=/usr/local/cuda-9.0`

## CuDNN

go to Nvidia website and download CuDNN v7.4.2 runtime deb

`sudo dpkg -i <cudnn-runtime>.deb`

## Conda

go to Anaconda website and download Python3.7 installer for linux

`bash ~/Downloads/Anaconda3-5.3.1-Linux-x86_64.sh`

## Setup conda environment

`conda create -n tf_cu90 python=3.6`

`pip install numpy`

## Tensorflow
```
sudo apt update
sudo apt install python3-dev python3-pip
pip install --upgrade tensorflow-gpu
```

verify: `python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"`

## Keras

`pip install keras`

## Setup CUDA on Conda

`mkdir -p ~/anaconda3/envs/tf_cu90/etc/conda/activate.d`

`subl ~/anaconda3/envs/tf_cu90/etc/conda/activate.d/activate.sh`

```
#!/bin/sh
ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64:/lib/nccl/cuda-9:$LD_LIBRARY_PATH
```

`chmod +x ~/anaconda3/envs/tf_cu90/etc/conda/deactivate.d/deactivate.sh`

`mkdir -p ~/anaconda3/envs/tf_cu90/etc/conda/deactivate.d`

`subl ~/anaconda3/envs/tf_cu90/etc/conda/deactivate.d/deactivate.sh`

```
#!/bin/sh
export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH
unset ORIGINAL_LD_LIBRARY_PATH
```

`chmod +x ~/anaconda3/envs/tf_cu90/etc/conda/deactivate.d/deactivate.sh`

## OpenCV

`sudo apt-get update`

```
sudo apt-get install build-essential cmake pkg-config libatlas-base-dev gfortran unzip
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3.6-dev
sudo apt install python-pip
```

```
wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
unzip opencv-3.4.0.zip
cd opencv-3.4.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D PYTHON_EXECUTABLE=~/.virtualenvs/r21d/bin/python \
        -D BUILD_EXAMPLES=ON \
	-D BUILD_SHARED_LIBS=ON ..
make -j8
sudo make install
sudo ldconfig
```

```
cd ~/anaconda3/envs/tf_cu90/lib/python3.6/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/<cv2.so> cv2.so
```
