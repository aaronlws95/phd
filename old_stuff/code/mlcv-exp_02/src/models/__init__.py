def get_model(cfg, mode, load_epoch):
    model = cfg['model']
    if model == 'yolov2_bbox':
        from src.models.yolov2_bbox import YOLOV2_Bbox
        return YOLOV2_Bbox(cfg, mode, load_epoch)
    elif model == 'hpo_ar':
        from src.models.hpo_ar import HPO_AR
        return HPO_AR(cfg, mode, load_epoch)
    elif model == 'hpo_hand':
        from src.models.hpo_hand import HPO_Hand
        return HPO_Hand(cfg, mode, load_epoch)
    elif model == 'hpo_bbox_ar':
        from src.models.hpo_bbox_ar import HPO_Bbox_AR
        return HPO_Bbox_AR(cfg, mode, load_epoch)
    elif model == 'hpo_bbox_ar_SL':
        from src.models.hpo_bbox_ar_SL import HPO_Bbox_AR_SL
        return HPO_Bbox_AR_SL(cfg, mode, load_epoch)
    elif model == 'multireso_hand_crop':
        from src.models.multireso_hand_crop import Multireso_Hand_Crop
        return Multireso_Hand_Crop(cfg, mode, load_epoch)
    elif model == 'multireso_hand_crop_normuvd':
        from src.models.multireso_hand_crop_normuvd import Multireso_Hand_Crop_Normuvd
        return Multireso_Hand_Crop_Normuvd(cfg, mode, load_epoch)
    elif model == 'yolo_hand_crop':
        from src.models.yolo_hand_crop import YOLO_Hand_Crop
        return YOLO_Hand_Crop(cfg, mode, load_epoch)
    elif model == 'cpm_hand_heatmap':
        from src.models.cpm_hand_heatmap import CPM_Hand_Heatmap
        return CPM_Hand_Heatmap(cfg, mode, load_epoch)
    else:
        raise Exception('{} is not a valid model'.format(model))