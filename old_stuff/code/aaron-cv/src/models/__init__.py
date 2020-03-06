from src.models.model                           import Model
from src.models.yolov2_fpha                     import YOLOV2_FPHA
from src.models.yolov2_fpha_reg                 import YOLOV2_FPHA_Reg
from src.models.yolov2_voc                      import YOLOV2_VOC
from src.models.yolov2_fpha_hpo_bbox            import YOLOV2_FPHA_HPO_Bbox
from src.models.fpha_hpo                        import FPHA_HPO
from src.models.znb_pose                        import ZNB_Pose
from src.models.znb_lift                        import ZNB_Lift
from src.models.znb_handseg                     import ZNB_Handseg
from src.models.multireso                       import Multireso
from src.models.multireso_from_pred             import Multireso_from_pred
from src.models.multireso_normuvd               import Multireso_Normuvd
from src.models.yolov2_fpha_hpo_bbox_2hand_0    import YOLOV2_FPHA_HPO_Bbox_2Hand_0
from src.models.yolov2_fpha_hpo_bbox_2hand_1    import YOLOV2_FPHA_HPO_Bbox_2Hand_1
from src.models.fpha_hpo_2hand_1                import FPHA_HPO_2Hand_1
from src.models.tsn_ek                          import TSN_EK
from src.models.tsn_1out                        import TSN_1Out
from src.models.tsn_ek_yolov2_bb                import TSN_EK_YOLOV2_BB
from src.models.tsn_1out_yolov2_bb              import TSN_1Out_YOLOV2_BB
from src.models.hpo_tsn_fpha                    import HPO_TSN_FPHA
from src.models.hand_paf                        import Hand_PAF
from src.models.fpha_hpo_action_noun            import FPHA_HPO_Action_Noun
from src.models.ek_hpo_action_noun              import EK_HPO_Action_Noun

def get_model(cfg, training, load_epoch, logger, tb_logger):
    name = cfg['model']
    args = (cfg, training, load_epoch, logger, tb_logger)
    if      name == 'yolov2_fpha_reg':
        return YOLOV2_FPHA_Reg(*args)
    elif    name == 'yolov2_fpha':
        return YOLOV2_FPHA(*args)
    elif    name == 'yolov2_voc':
        return YOLOV2_VOC(*args)
    elif    name == 'yolov2_fpha_hpo_bbox':
        return YOLOV2_FPHA_HPO_Bbox(*args)
    elif    name == 'yolov2_fpha_hpo_bbox_2hand_0':
        return YOLOV2_FPHA_HPO_Bbox_2Hand_0(*args)
    elif    name == 'yolov2_fpha_hpo_bbox_2hand_1':
        return YOLOV2_FPHA_HPO_Bbox_2Hand_1(*args)
    elif    name == 'fpha_hpo':
        return FPHA_HPO(*args)
    elif    name == 'fpha_hpo_2hand_1':
        return FPHA_HPO_2Hand_1(*args)
    elif    name == 'znb_pose':
        return ZNB_Pose(*args)
    elif    name == 'znb_lift':
        return ZNB_Lift(*args)
    elif    name == 'znb_handseg':
        return ZNB_Handseg(*args)
    elif    name == 'multireso':
        return Multireso(*args)
    elif    name == 'multireso_from_pred':
        return Multireso_from_pred(*args)
    elif    name == 'multireso_normuvd':
        return Multireso_Normuvd(*args)
    elif    name == 'tsn_ek':
        return TSN_EK(*args)
    elif    name == 'tsn_1out':
        return TSN_1Out(*args)
    elif    name == 'tsn_ek_yolov2_bb':
        return TSN_EK_YOLOV2_BB(*args)
    elif    name == 'tsn_1out_yolov2_bb':
        return TSN_1Out_YOLOV2_BB(*args)
    elif    name == 'hpo_tsn_fpha':
        return HPO_TSN_FPHA(*args)
    elif    name == 'hand_paf':
        return Hand_PAF(*args)
    elif name == 'fpha_hpo_action_noun':
        return FPHA_HPO_Action_Noun(*args)
    elif name == 'ek_hpo_action_noun':
        return EK_HPO_Action_Noun(*args)

    ###################################
    elif    name == 'exp_mixweights':
        from src.models.exp_mixweights import EXP_MIXWEIGHTS
        return EXP_MIXWEIGHTS(*args)
    elif    name == 'exp_mixweights2':
        from src.models.exp_mixweights2 import EXP_MIXWEIGHTS2
        return EXP_MIXWEIGHTS2(*args)    
    else:
        raise ValueError(f'{name} is not a valid model')