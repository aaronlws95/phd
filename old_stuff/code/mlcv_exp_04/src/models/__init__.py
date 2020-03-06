from src.models.hpo_hand.hpo_hand_model import HPO_Hand_Model
from src.models.hpo_hand_object.hpo_hand_object_model import HPO_Hand_Object_Model
from src.models.yolov2_bbox.yolov2_bbox_model import YOLOV2_Bbox_Model
from src.models.hand_crop_disc.hand_crop_disc_model import Hand_Crop_Disc_Model
models = {}
models['hpo_hand'] = HPO_Hand_Model
models['hpo_hand_object'] = HPO_Hand_Object_Model
models['yolov2_bbox'] = YOLOV2_Bbox_Model
models['hand_crop_disc'] = Hand_Crop_Disc_Model

def get_model(cfg):
    m_name = cfg['model']
    model = models[m_name](cfg)
    return model
