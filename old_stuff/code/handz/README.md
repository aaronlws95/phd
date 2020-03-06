# Handz 

[PyTorch](https://pytorch.org/) framework for deep learning. I'm trying to consolidate all my scripts and files for deep learning in this repo so that it is easy to access. The idea is to have a unified framework for training and evaluating my deep learning models.

## Setup

1. Create conda environment

`conda env create -f=environment.yml`

2. Update `dir.json` with directory paths and move to `config` folder

```
mkdir config
mv dir.json config
```

## Usage

**Preparation**

* Download corresponding dataset

* Refer to corresponding model setup

**Training**

* `python train.py --conf config/<config.json>`

**Predicting**

* `python predict.py --conf config/<config.json> --epoch <epoch>`

**Evaluation**

* Refer to corresponding notebook

## Debug

`sh debug/train_predict_debug/train_predict_debug.sh`
** You need to update the directories (kinda annoying)

## Add new model

1. Data preprocessing
    * prepare_data scripts must be called from within the folder `cd prepare_data`
2. Data loader
3. Model network
4. Loss calculation
5. Validation
6. Training details
7. Predicting
8. Evaluation

# Models

| Model | Status   | Result | 
|:--------:|:-------:|:-------:|
| [HPO](#HPO) | Improving   | -     |      
| [YOLOV2_VOC](#YOLOV2_VOC) | Does not match original | 71.7 mAP |  
| [YOLOV2_FPHA](#YOLOV2_FPHA) | Suitable | 0.87 IOU   |   
| [YOLOV2_FPHA_reg](#YOLOV2_FPHA_reg) | Training | 0.64 AUC   |  
| [YOLOV2_FPHA_HPOreg](#YOLOV2_FPHA_HPOreg) | Training | -   |  
| [YOLOV3_COCO](#YOLOV3_COCO) | Training | 41.0 mAP  |  
| [YOLOV3_VOC](#YOLOV3_VOC) | Bad | 47.0 mAP   |  
| [YOLOV3_FPHA](#YOLOV3_FPHA) | Untested | -   |  

-------------------------------------------------------------

## HPO

HPO solution for hand pose estimation. Based on [singleshotpose](https://github.com/Microsoft/singleshotpose) implementation.

**Model**: HPOModel

**Network**: Darknet  

* output shape: (batch_size, 320, width/32, width/32)
    * 320 = depth*(no. keypoints + conf) = 5*64

**Data Loader**: HPODataset

* dataset: [First-Person Hand Action](https://github.com/guiggh/hand_pose_action) (FPHA)

* labels: normalized 21 3D joint positions (21,3)

**Loss**: HPOLoss

-------------------------------------------------------------

## YOLOV2Model_VOC

Multi-class multi bounding box object detection with YOLOv2. Based on https://github.com/marvis/pytorch-yolo2. 

**Model**: YOLOV2_VOC

* get Pascal VOC data

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

* generate labels

```
cd prepare_data
python prepare_data/voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```

**Network**: Darknet 

* pretrain: darknet19_448.conv.23 pretrained on ImageNet

`wget http://pjreddie.com/media/files/darknet19_448.conv.23`

* cfg: yolov2-voc-1class.cfg

* output shape: (batch_size, 25, width/32, height/32)

**Data Loader**: YOLODataset_FPHA 

* dataset: [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

* labels: normalized bounding boxes [cls_id, x_cen, y_cen, w, h]

**Loss**: RegionLoss_1Class 

**Training**

* Learning rate steps are per batch (100 steps = 100 batches). Need to update steps if you change the batch size.

-------------------------------------------------------------

## YOLOV2_FPHA

Hand bounding box detection with YOLOv2 on FPHA dataset. Based on https://github.com/marvis/pytorch-yolo2. 

**Setup**

* prepare annotations

```
cd prepare_data
python prepare_data/create_lmdb_annot_YOLO_FPHA.py
```

* get anchors (optional)

```
cd prepare_data
python prepare_data/get_kmeans_anchors_YOLO_FPHA.py
```

**Model**: YOLOV2Model_1Class

**Network**: Darknet 

* pretrain: darknet19_448.conv.23 pretrained on ImageNet

* cfg: yolov2-voc-1class.cfg

* output shape: (batch_size, 25, width/32, height/32)

**Data Loader**: YOLODataset_FPHA 

* dataset: FPHA

* labels: normalized bounding boxes [cls_id, x_cen, y_cen, w, h]

**Loss**: RegionLoss_1Class 

-------------------------------------------------------------

## YOLOV2_FPHA_reg

Hand bounding box detection with YOLOv2 on FPHA dataset with hand pose estimation sub-task. Based on https://github.com/marvis/pytorch-yolo2. 

**Model**: YOLOV2Model_1Class_reg

**Network**: Darknet_reg 

* pretrain: YOLOV2_FPHA base model trained to 100 epochs 

* cfg: yolov2-voc-1class-reg.cfg

* output shape: (batch_size, 25, width/32, height/32), (63) 
    * 25=num_anchors*(x,y,w,h,conf)

**Data Loader**: YOLODataset_FPHA_reg 

* dataset: FPHA

* labels: normalized bounding boxes [cls_id, x_cen, y_cen, w, h], normalized 21 3D joint positions

**Loss**: RegionLoss_1Class_reg 

-------------------------------------------------------------

## YOLOV2_FPHA_HPOreg

Hand bounding box detection with YOLOv2 on FPHA dataset with hand pose estimation sub-task. Based on https://github.com/marvis/pytorch-yolo2. 

**Model**: YOLOV2Model_1Class_HPOreg

**Network**: Darknet_reg 

* pretrain: YOLOV2_FPHA base model trained to 100 epochs 

* cfg: yolov2-voc-1class-hporeg.cfg

* output shape: (batch_size, 25, width/32, height/32), (batch_size, 320, width/32, height/32)

**Data Loader**: YOLODataset_FPHA_reg 

* dataset: FPHA

* labels: normalized bounding boxes [cls_id, x_cen, y_cen, w, h], normalized 21 3D joint positions

**Loss**: RegionLoss_1Class, HPOLoss 

-------------------------------------------------------------

## YOLOV3_COCO

Multi-class multi bounding box object detection with YOLOv3 on COCO dataset. Based on https://github.com/ultralytics/yolov3. 

**Setup**

* prepare annotations

```
cd prepare_data
sh prepare_data/get_coco_dataset.sh
```

**Model**: YOLOV3Model_COCO

**Network**: Darknet V3

* pretrain: darknet53.conv.74 pretrained on ImageNet

* cfg: yolov3.cfg

* output shape: [(batch_size, 3, 85, 13, 13), (batch_size, 3, 85, 26, 26), (batch_size, 3, 85, 52, 52)] 
    * 3=num_anchors, 85=num_class+(x,y,w,h,conf)

**Data Loader**: YOLOV3Dataset_COCO 

* dataset: COCO

* labels: normalized bounding boxes [image_idx, cls_id, x_cen, y_cen, w, h]

**Loss**: YOLOV3Loss

-------------------------------------------------------------

## YOLOV3_VOC

Multi-class multi bounding box object detection with YOLOv3 on VOC dataset. Based on https://github.com/ultralytics/yolov3.

**Model**: YOLOV3Model_VOC

**Network**: Darknet V3

* pretrain: darknet53.conv.74 pretrained on ImageNet

* cfg: yolov3-voc.cfg

* output shape: [(batch_size, 3, 85, 13, 13), (batch_size, 3, 85, 26, 26), (batch_size, 3, 85, 52, 52)] 

**Data Loader**: YOLOV3Dataset_VOC 

* dataset: VOC

* labels: normalized bounding boxes [image_idx, cls_id, x_cen, y_cen, w, h]

**Loss**: YOLOV3Loss

-------------------------------------------------------------

## YOLOV3_FPHA

Bounding box hand detection with YOLOv3 on FPHA dataset. Based on https://github.com/ultralytics/yolov3.

**Model**: YOLOV3Model_1Class

**Network**: Darknet V3

* pretrain: darknet53.conv.74 pretrained on ImageNet

* cfg: yolov3-1class.cfg

* output shape: [(batch_size, 3, 5, 13, 13), (batch_size, 3, 5, 26, 26), (batch_size, 3, 5, 52, 52)] 

**Data Loader**: YOLOV3Dataset_FPHA

* dataset: FPHA

* labels: normalized bounding boxes [image_idx, x_cen, y_cen, w, h]

**Loss**: YOLOV3Loss_1Class

-------------------------------------------------------------

# References

* [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/abs/1711.08848)

* [First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations](https://arxiv.org/pdf/1704.02463)

* [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)

* [YOLO9000: Better, Faster, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf)

* [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

* [The PASCAL Visual Object Classes (VOC) Challenge](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)