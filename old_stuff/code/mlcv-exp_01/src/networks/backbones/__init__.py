# all backbones takes as input: (B, C, H, W)
def get_backbone(cfg):
    bb = cfg['backbone']
    if bb == 'yolov2':
        from src.networks.backbones.yolov2 import YOLOV2
        backbone = YOLOV2(cfg)
    else:
        raise Exception('{} is not a valid backbone'.format(bb))
    
    if cfg['freeze_bb']:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone 

def get_stride(cfg):
    bb = cfg['backbone']
    if bb == 'yolov2':
        return 32
    else:
        raise Exception('{} is not a valid backbone'.format(bb))