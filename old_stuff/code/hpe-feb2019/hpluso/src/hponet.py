from keras.models import Model, Input, load_model
from keras.optimizers import SGD

import constants
import layers as l
from loss import hpo_loss

def hponet_base():
    #as specified in supplementary paper
    #based on darknet yolo
    input_image = Input(shape=(constants.RSZ_HEIGHT, constants.RSZ_WIDTH, constants.CHANNEL))
    al = [input_image]
    x = l.conv(input_image, 32, 3, al) #0
    x = l.max_pool(x, al)           #1
    x = l.conv(x, 64, 3, al)           #2
    x = l.max_pool(x, al)           #3
    x = l.conv(x, 128, 3, al)          #4
    x = l.conv(x, 64, 1, al)        #5
    x = l.conv(x, 128, 3, al)          #6
    x = l.max_pool(x, al)           #7
    x = l.conv(x, 256, 3, al)          #8
    x = l.conv(x, 128, 1, al)       #9
    x = l.conv(x, 256, 3, al)          #10
    x = l.max_pool(x, al)           #11
    x = l.conv(x, 512, 3, al)          #12
    x = l.conv(x, 256, 1, al)       #13
    x = l.conv(x, 512, 3, al)          #14
    x = l.conv(x, 256, 1, al)       #15
    x = l.conv(x, 512, 3, al)          #16
    x = l.max_pool(x, al)           #17
    x = l.conv(x, 1024, 3, al)         #18
    x = l.conv(x, 512, 1, al)       #19
    x = l.conv(x, 1024, 3, al)         #20
    x = l.conv(x, 512, 1, al)       #21
    x = l.conv(x, 1024, 3, al)         #22
    x = l.conv(x, 1024, 3, al)         #23
    x = l.conv(x, 1024, 3, al)         #24
    x = l.route([-9], al)           #25
    x = l.conv(x, 64, 3, al)           #26
    x = l.reorg(x, al)              #27
    x = l.route([-1, -4], al)       #28
    x = l.conv(x, 1024, 3, al)         #29
    # x = l.conv(all_layers, x, 720, 1)     #30 13x13x(10x(3xNc+1+Na+No))
    model = Model(inputs=al[0], outputs=x)
    return model

def hponet_hpe():
    #model for HPE only
    base_model = hponet_base()
    x = l.conv(base_model.output, 320, 1)       #30 13x13x(5x(3xNc+1))
    x = l.offset_to_uvz(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer = SGD(lr = 0.0001), loss = hpo_loss)
    return model

def hponet_hpe_yad2k():
    #model for HPE only with pretrained yolov2
    base_model = load_model('../data/yolov2_hpo_hpe.h5')
    x = l.offset_to_uvz(base_model.output)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer = SGD(lr = 0.0001), loss = hpo_loss)
    # model.summary()
    return model

def hponet_hpe_yad2k_noconf_noofs():
    #model for HPE only with pretrained yolov2
    #no confidence prediction
    base_model = load_model('../data/yolov2_hpo_hpe_noconf.h5')
    x = l.reshape(base_model.output)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer = SGD(lr = 0.0001), loss = 'mean_squared_error')
    return model




