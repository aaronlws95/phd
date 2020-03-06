def get_model(conf, device, load_epoch, is_train, exp_dir, deterministic, logger=None):
    name = conf["name"]
    if name == "HPOnet":
        from .HPOModel import HPOModel
        model = HPOModel(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model
    elif name == "YOLOV2net_VOC":
        from .YOLOV2Model import YOLOV2Model_VOC
        model = YOLOV2Model_VOC(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model
    elif name == "YOLOV2net_1Class":
        from .YOLOV2Model import YOLOV2Model_1Class
        model = YOLOV2Model_1Class(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model
    elif name == "YOLOV2net_1Class_reg":
        from .YOLOV2Model import YOLOV2Model_1Class_reg
        model = YOLOV2Model_1Class_reg(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model    
    elif name == "YOLOV2net_1Class_HPOreg":
        from .YOLOV2Model import YOLOV2Model_1Class_HPOreg
        model = YOLOV2Model_1Class_HPOreg(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model        
    elif name == "YOLOV3net_COCO":
        from .YOLOV3Model import YOLOV3Model_COCO
        model = YOLOV3Model_COCO(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model    
    elif name == "YOLOV3net_VOC":
        from .YOLOV3Model import YOLOV3Model_VOC
        model = YOLOV3Model_VOC(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model     
    elif name == "YOLOV3net_1Class":
        from .YOLOV3Model import YOLOV3Model_1Class
        model = YOLOV3Model_1Class(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model        
    elif name == "Multiresonet":
        from .MultiresoModel import MultiresoModel
        model = MultiresoModel(conf, device, load_epoch, is_train, exp_dir, deterministic, logger)
        return model
    else: 
        raise ValueError(f"{name} is not a valid model")