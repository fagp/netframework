from .lossfunc import *
import torch.nn as nn
import json
from utils.utils import Decoder

def selectloss(lossname,parameter={},use_cuda=False):
    print('Selecting loss function: ',lossname)
    if lossname=='ce':
        criterion = nn.CrossEntropyLoss(**parameter)
        criterioneval="(outputs[0],sample['label'][:,0])"
    elif lossname=='mse':
        criterion = nn.MSELoss(**parameter)
        criterioneval="(outputs[0][:,0],sample['label'][:,0].float())"
    elif lossname=='l1':
        criterion = nn.L1Loss(**parameter)
        criterioneval="(outputs[0],sample['label'].float())"
    elif lossname=='accuracy':
        criterion = Accuracy()
        criterioneval="(outputs[0],sample['label'][:,0])"
    else:
        raise 'Loss '+lossname+' not available'

    if torch.cuda.is_available() and use_cuda:
        criterion=criterion.cuda()

    return criterion,criterioneval

def get_metric_path(config_file='defaults/metrics.json'):
    metrics = json.load(open(config_file),cls=Decoder)
    return metrics
