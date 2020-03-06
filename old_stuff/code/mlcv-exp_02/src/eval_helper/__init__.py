def get_eval_helper(cfg, epoch, split):
    model = cfg['model']

    if model == 'yolov2_bbox':
        from src.eval_helper.yolov2_bbox_helper import YOLOV2_Bbox_Helper
        return YOLOV2_Bbox_Helper(cfg, epoch, split)
    elif model == 'hpo_ar':
        from src.eval_helper.hpo_ar_helper import HPO_AR_Helper
        return HPO_AR_Helper(cfg, epoch, split)
    elif model == 'hpo_hand':
        from src.eval_helper.hpo_hand_helper import HPO_Hand_Helper
        return HPO_Hand_Helper(cfg, epoch, split)
    elif model == 'hpo_bbox_ar' or model == 'hpo_bbox_ar_SL':
        from src.eval_helper.hpo_bbox_ar_helper import HPO_Bbox_AR_Helper
        return HPO_Bbox_AR_Helper(cfg, epoch, split)
    elif model == 'multireso_hand_crop':
        from src.eval_helper.multireso_hand_crop_helper import Multireso_Hand_Crop_Helper
        return Multireso_Hand_Crop_Helper(cfg, epoch, split)
    elif model == 'multireso_hand_crop_normuvd':
        from src.eval_helper.multireso_hand_crop_normuvd_helper import Multireso_Hand_Crop_Normuvd_Helper
        return Multireso_Hand_Crop_Normuvd_Helper(cfg, epoch, split)
    elif model == 'yolo_hand_crop':
        from src.eval_helper.yolo_hand_crop_helper import YOLO_Hand_Crop_Helper
        return YOLO_Hand_Crop_Helper(cfg, epoch, split)
    elif model == 'cpm_hand_heatmap':
        from src.eval_helper.cpm_hand_heatmap_helper import CPM_Hand_Heatmap_Helper
        return CPM_Hand_Heatmap_Helper(cfg, epoch, split)
    else:
        raise Exception('{} is not a valid model'.format(model))