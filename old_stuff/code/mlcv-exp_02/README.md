# mlcv-exp: Machine Learning Computer Vision Experiments

## Setup
```bash
./setup/init.sh /path/to/root/
```

### Root directory
```bash
Root
├── mlcv-exp
│   ├── data
│   │   ├── cfg
│   │   │   ├── model1
│   │   │   └── model2
│   │   ├── exp
│   │   │   ├── model1
│   │   │   └── model2
│   │   ├── labels
│   │   ├── weights
│   │   └── root.txt
│   ├── setup
│   ├── test
│   ├── main.py
│   └── src
├── dataset1
├── dataset2
└── dataset3
```

## Usage

* Training
```bash
python main.py --cfg <cfgfile> --mode train --epoch <optional>
```

* Inference
```bash
python main.py --cfg <cfgfile> --mode test --epoch <load_epoch> --split <optional>
```

* Evaluation

Use [evaluation.ipynb](notebooks/evaluation.ipynb) to evaluate models after inference

# Implementation

| Model | Datasets | Ref | Note |
|-|-|-|-|
| [yolov2_bbox](src/models/yolov2_bbox.py) | [fpha_bbox](src/models/fpha_bbox_dataset.py) | [link](https://github.com/marvis/pytorch-yolo2) | Implemented |
| [hpo_ar](src/models/hpo_ar.py) | [fpha_bbox](src/models/hpo_ar_dataset.py) | [link](https://arxiv.org/pdf/1904.05349.pdf) | Implemented |
| [hpo_hand](src/models/hpo_hand.py) | [fpha_bbox](src/models/hpo_hand_dataset.py) | [link](https://arxiv.org/pdf/1904.05349.pdf) | Implemented |
| [hpo_bbox_ar](src/models/hpo_bbox_ar.py) | [fpha_bbox](src/models/hpo_bbox_ar_dataset.py) | [link](https://arxiv.org/pdf/1904.05349.pdf) | Implemented |

| Backbone | Ref |
|-|-|
| [yolov2](src/networks/yolov2.py) | [link](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg)
