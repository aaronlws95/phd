# fpha

For setting up environment see SETUP.MD

## qi2016

Code for [[Qi, 2016]](https://arxiv.org/abs/1604.03334): Spatial Attention Deep Net with Partial PSO for Hierarchical Hybrid Hand Pose Estimation

### huawei

Code in Keras

`data`: Model weights, sample images 

`Hier_Estimator/test_pose_estimator.py`: Run the model

`notebooks/`: Jupyter notebooks

### fpha

Adapted code in Keras for training on FPHA dataset

Steps:

```
1. prepare_data.py
2. train.py
3. predict.py
4. evaluation.ipynb
``` 

## hpluso

Code for H+O

Get pretrained yolov2 model from [https://github.com/allanzelener/YAD2K](https://github.com/allanzelener/YAD2K):
```
wget http://pjreddie.com/media/files/yolov2.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg
./yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
```

## xinghaochen

Evaluation scripts from [https://github.com/xinghaochen/awesome-hand-pose-estimation](https://github.com/xinghaochen/awesome-hand-pose-estimation)

Requires Python2.7

