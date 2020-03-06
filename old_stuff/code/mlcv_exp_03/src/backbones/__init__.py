from src.backbones.yolov2 import YOLOV2
from src.backbones.resnet50 import ResNet50
backbones = {}
backbones['yolov2'] = YOLOV2
backbones['resnet50'] = ResNet50

# backbones are networks with input: (B, C, H, W)
def get_backbone(cfg):
    bb_name = cfg['backbone']
    backbone = backbones[bb_name](cfg)
    if cfg['freeze_bb']:
        backbone.freeze()
    return backbone