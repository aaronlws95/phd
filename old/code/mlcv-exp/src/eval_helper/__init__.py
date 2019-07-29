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
    else:
        raise Exception('{} is not a valid model'.format(model))