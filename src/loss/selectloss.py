from .lossfunc import *
import torch.nn as nn
import json
from utils.utils import Decoder

def selectloss(lossname,parameter={},use_cuda=False, config_file='defaults/loss_definition.json'):
    loss_func=get_loss_path(lossname,config_file)
    criterion = eval(loss_func['criterion']+'(**parameter)')
    criterionparam = loss_func['criterionparam']

    if torch.cuda.is_available() and use_cuda:
        criterion=criterion.cuda()

    return criterion,criterionparam

def get_metric_path(config_file='defaults/metrics.json'):
    metrics = json.load(open(config_file),cls=Decoder)
    return metrics

def get_loss_path(name, config_file='defaults/loss_definition.json'):
    loss_func = json.load(open(config_file),cls=Decoder)
    if name not in loss_func:
        raise 'Function '+name+' not found in '+config_file
    return loss_func[name]