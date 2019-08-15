from src.eval_helper.hpo_hand_helper import HPO_Hand_Helper
from src.eval_helper.hpo_hand_object_helper import HPO_Hand_Object_Helper
from src.eval_helper.yolov2_bbox_helper import YOLOV2_Bbox_Helper 
eval_helpers = {}
eval_helpers['hpo_hand'] = HPO_Hand_Helper
eval_helpers['hpo_hand_object'] = HPO_Hand_Object_Helper
eval_helpers['yolov2_bbox'] = YOLOV2_Bbox_Helper

def get_eval_helper(cfg, epoch, split):
    model = cfg['model']
    eval_helper = eval_helpers[model](cfg, epoch, split)
    return eval_helper