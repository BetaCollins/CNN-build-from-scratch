from layers import *
from models import Model

def MNISTNet():
    conv1_params={
        'kernel_h': 3,
        'kernel_w': 3,
        'pad': 0,
        'stride': 1,
        'in_channel': 1,
        'out_channel': 6
    }
    conv2_params={
        'kernel_h': 3,
        'kernel_w': 3,
        'pad': 0,
        'stride': 1,
        'in_channel': 6,
        'out_channel': 16
    }
    pool1_params={
        'pool_type': 'max',
        'pool_height': 2,
        'pool_width': 2,
        'stride': 2,
        'pad': 0
    }
    pool2_params={
        'pool_type': 'max',
        'pool_height': 3,
        'pool_width': 3,
        'stride': 2,
        'pad': 0
    }
    model = Model()
    model.add(Convolution(conv1_params, name='conv1', initializer=Guassian(std=0.001)))
    model.add(ReLU(name='relu1'))
    model.add(Pooling(pool1_params, name='pooling1'))
    model.add(Convolution(conv2_params, name='conv2', initializer=Guassian(std=0.001)))
    model.add(ReLU(name='relu2'))
    model.add(Pooling(pool2_params, name='pooling2'))
    # model.add(Dropout(ratio=0.25, name='dropout1'))
    model.add(Flatten(name='flatten'))
    model.add(FCLayer(400, 256, name='fclayer1', initializer=Guassian(std=0.01))) 
    model.add(ReLU(name='relu3'))
    # model.add(Dropout(ratio=0.5))
    model.add(FCLayer(256, 10, name='fclayer2', initializer=Guassian(std=0.01)))
    return model