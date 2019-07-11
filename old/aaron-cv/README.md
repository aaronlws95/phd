# aaron-cv
Computer Vision Machine Learning platform for experimentation in PyTorch. Experimental results can be viewed [here](https://docs.google.com/spreadsheets/d/1ZVQM1k-jXOoBonEUhJQsqcFRF_vkezHMOKDMq-EVPxY/edit?usp=sharing).

# Setup

* Setup environment
```
git clone https://github.com/aaronlws95/aaron-cv.git
conda env create -f environment.yml
```

* Create directory reference
```
printf "/path/to/data/" >> src/utils/data_dir.txt
```
All relevant data folders should be located in the path


# Usage

* Training

```
python run.py --cfg <cfgfile> --mode train --epoch <optional>
```

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2345 run.py --cfg <cfgfile> --mode train --epoch <optional>
```

* Testing

```
python run.py --cfg <cfgfile> --mode test --epoch <load_epoch> --split <optional>
```

# Implemented

| Model                                  | Network                                    | Loss                       | Dataset                            |
|----------------------------------------|--------------------------------------------|----------------------------|------------------------------------|
| [yolov2_voc](src/models/yolov2_voc.py) | [yolov2_voc_net](net_cfg/yolov2_voc_net.cfg) | [region](src/loss/region.py) | [voc_bbox](src/datasets/voc_bbox.py) |
| [yolov2_fpha](src/models/yolov2_fpha.py) | [yolov2_fpha_net](net_cfg/yolov2_fpha_net.cfg) | [region_noclass_1bbox](src/loss/region_noclass_1bbox.py) | [fpha_bbox](src/datasets/fpha_bbox.py) | 
| [yolov2_fpha_reg](src/models/yolov2_fpha_reg.py) | [yolov2_fpha_reg_net](net_cfg/yolov2_fpha_reg_net.cfg) | [region_noclass_1bbox](src/loss/region_noclass_1bbox.py) + MSE | [fpha_bbox_hand](src/datasets/fpha_bbox_hand.py) | 
| [yolov2_fpha_hpo_bbox](src/models/yolov2_fpha_hpo_bbox.py) | [yolov2_fpha_hpo_bbox_net](net_cfg/yolov2_fpha_hpo_bbox_net.cfg) | [region_noclass_1bbox](src/loss/region_noclass_1bbox.py) + [hpo](src/loss/hpo.py)| [fpha_bbox_hand](src/datasets/fpha_bbox_hand.py) |
| [fpha_hpo](src/models/fpha_hpo.py) | [fpha_hpo_net](net_cfg/fpha_hpo_net.cfg) | [hpo](src/loss/hpo.py)| [fpha_hand](src/datasets/fpha_hand.py) |
| [znb_pose](src/models/znb_pose.py) | [znb_pose_net](net_cfg/znb_pose_net.cfg) | L2 | [rhd_smap](src/datasets/rhd_smap.py) |
| [znb_lift](src/models/znb_lift.py)| [znb_pose_prior_net](net_cfg/znb_pose_prior_net.cfg) + [znb_viewpoint_net](net_cfg/znb_viewpoint_net.cfg) | MSE | [rhd_smap_canon](src/datasets/rhd_smap_canon.py) |
|  [znb_handseg](src/models/znb_handseg.py) | [znb_handseg_net](net_cfg/znb_handseg_net.cfg) | BCEWithLogitsLoss | [rhd_mask](src/datasets/rhd_mask.py) |
|  [multireso](src/models/multireso.py) | multireso_net  | MSE | [fpha_multireso_crop_hand](src/datasets/fpha_multireso_crop_hand.py) |
|  [tsn](src/models/tsn_1out.py) | TSN  | CrossEntropyLoss | [tsn_labels](src/datasets/tsn_labels.py) |

** other models are experimental variations

# Tips

* [Notebooks](notebooks) for visualization and evaluation
* Training checklist
    - [ ] Does the model overfit?
    - [ ] Does the data loader output look correct?
    - [ ] Is the data configuration correct?
    - [ ] Is the save directory correct?
* Pascal VOC preparation
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
python prepare_data/voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
* Epic Kitchen preparation
```
# Create directories
find . -name "*.tar" | xargs -P 5 -I fileName sh -c 'mkdir $(dirname "fileName")/$(basename -s .tar "fileName")'
# Recursively tar
find . -name "*.tar" | xargs -P 5 -I fileName sh -c 'tar xvf "fileName" -C "$(dirname "fileName")/$(basename -s .tar "fileName")"'
```

# References

## Code
* https://github.com/marvis/pytorch-yolo2
* https://github.com/ultralytics/yolov3
* https://github.com/lmb-freiburg/hand3d
* https://github.com/yjxiong/tsn-pytorch

## Papers
* [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/abs/1711.08848)
* [First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations](https://arxiv.org/pdf/1704.02463)
* [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)
* [YOLO9000: Better, Faster, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
* [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
* [The PASCAL Visual Object Classes (VOC) Challenge](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)
* [Learning to Estimate 3D Hand Pose from Single RGB Images](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zimmermann_Learning_to_Estimate_ICCV_2017_paper.pdf)
* [Temporal Segment Networks for Action Recognition in Videos](https://arxiv.org/pdf/1705.02953.pdf)
