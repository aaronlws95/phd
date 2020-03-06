#!/bin/bash

curdir="${0%/*}"

python "$curdir"/../../train.py --conf "$curdir"/HPO_debug.json --enter 0
python "$curdir"/../../predict.py --conf "$curdir"/HPO_debug.json --epoch 5 --enter 0

python "$curdir"/../../train.py --conf "$curdir"/YOLO_VOC_debug.json --enter 0
python "$curdir"/../../predict.py --conf "$curdir"/YOLO_VOC_debug.json --epoch 5 --enter 0

python "$curdir"/../../train.py --conf "$curdir"/YOLO_FPHA_debug.json --enter 0
python "$curdir"/../../predict.py --conf "$curdir"/YOLO_FPHA_debug.json --epoch 5 --enter 0

python "$curdir"/../../train.py --conf "$curdir"/YOLO_FPHA_reg_debug.json --enter 0
python "$curdir"/../../predict.py --conf "$curdir"/YOLO_FPHA_reg_debug.json --epoch 5 --enter 0

python "$curdir"/../../train.py --conf "$curdir"/YOLO_FPHA_HPOreg_debug.json --enter 0
python "$curdir"/../../predict.py --conf "$curdir"/YOLO_FPHA_HPOreg_debug.json --epoch 5 --enter 0

python "$curdir"/../../train.py --conf "$curdir"/YOLOV3_COCO_debug.json --enter 0
python "$curdir"/../../predict.py --conf "$curdir"/YOLOV3_COCO_debug.json --epoch 5 --enter 0

python "$curdir"/../../train.py --conf "$curdir"/YOLOV3_VOC_debug.json --enter 0
python "$curdir"/../../predict.py --conf "$curdir"/YOLOV3_VOC_debug.json --epoch 5 --enter 0

python "$curdir"/../../train.py --conf "$curdir"/YOLOV3_FPHA_debug.json --enter 0
python "$curdir"/../../predict.py --conf "$curdir"/YOLOV3_FPHA_debug.json --epoch 5 --enter 0
