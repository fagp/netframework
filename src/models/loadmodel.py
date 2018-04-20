from .arch.vgg import *
from .arch.dpn import *
from .arch.lenet import *
from .arch.senet import *
from .arch.resnet import *
from .arch.resnext import *
from .arch.densenet import *
from .arch.googlenet import *
from .arch.mobilenet import *
from .arch.shufflenet import *
from .arch.preact_resnet import *

#from utils.utils import loadnet
import json
import torch
from torch import nn
from torch.nn import init
import torch.backends.cudnn as cudnn


def loadmodel(modelname,experimentparams,use_cuda=False,use_parallel=False,config_file='defaults/modelconfig.json'):
    print('Loading model: ',modelname)

    model_props = get_model_path(name=modelname, config_file=config_file)

    arch=model_props['arch']
    model_props.pop('arch',None)

    for key,value in experimentparams.items():
        model_props[key]=value

    modelparams=arch+'('
    for key,value in model_props.items():
        modelparams=modelparams+key+'=\''+value+'\','
    modelparams=modelparams[:-1]+')'
    
    net = eval(modelparams)

    if torch.cuda.is_available() and use_cuda:
        net=net.cuda()

    if use_cuda and use_parallel:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    init_params(net)

    return net,arch

def get_model_path(name, config_file='defaults/modelconfig.json'):
    model_config = json.load(open(config_file))
    if name not in model_config:
        raise 'Model '+name+' not found in '+config_file
    return model_config[name]

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight, gain=1)
            if m.bias is not None:
                init.constant(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)

        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)