import torch

def get_dataset(conf, train_mode, model, deterministic):
    name = conf["name"]
    if name == "HPO_FPHA":
        from .HPODataset import HPODataset_FPHA
        return HPODataset_FPHA(conf, train_mode, model, deterministic)
    elif name == "YOLO_VOC":
        from .YOLOV2Dataset import YOLOV2Dataset_VOC
        return YOLOV2Dataset_VOC(conf, train_mode, model, deterministic)
    elif name == "YOLO_FPHA":
        from .YOLOV2Dataset import YOLOV2Dataset_FPHA
        return YOLOV2Dataset_FPHA(conf, train_mode, model, deterministic)    
    elif name == "YOLO_FPHA_reg":
        from .YOLOV2Dataset import YOLOV2Dataset_FPHA_reg
        return YOLOV2Dataset_FPHA_reg(conf, train_mode, model, deterministic)            
    elif name == "YOLOV3_COCO":
        from .YOLOV3Dataset import YOLOV3Dataset_COCO
        return YOLOV3Dataset_COCO(conf, train_mode, model, deterministic) 
    elif name == "YOLOV3_VOC":
        from .YOLOV3Dataset import YOLOV3Dataset_VOC
        return YOLOV3Dataset_VOC(conf, train_mode, model, deterministic)         
    elif name == "YOLOV3_FPHA":
        from .YOLOV3Dataset import YOLOV3Dataset_FPHA
        return YOLOV3Dataset_FPHA(conf, train_mode, model, deterministic)                
    else: 
        raise ValueError(f"{name} is not a valid dataset")
    return net

def get_dataloader(conf, dataset, sampler, device, train_mode, deterministic=False, logger=None):   
    if not train_mode:
        conf["shuffle"] = False
        conf["num_workers"] = 2
        conf["aug"] = False
        sampler = None
    
    if logger:
        logger.log('-------------------')
        logger.log('CREATED DATA LOADER')
        logger.log('-------------------')
        for key, val in conf.items():
            if key == 'len':
                logger.log(key.upper() + ': ' + str(len(dataset)))
            else:
                logger.log(key.upper() + ': ' + str(val))
            
    if device != "cpu":
        kwargs = {'num_workers': conf["num_workers"], 'pin_memory': True}
    else:
        kwargs = {}     
        
    if hasattr(dataset, 'collate_fn'):
        return torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=conf["batch_size"],
                                            shuffle=conf["shuffle"],
                                            sampler=sampler,
                                            collate_fn=dataset.collate_fn,
                                            **kwargs,)               
    else:
        return torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=conf["batch_size"],
                                            shuffle=conf["shuffle"],
                                            sampler=sampler,
                                            **kwargs,)

