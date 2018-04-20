import json
from torch.utils import data
import torchvision.transforms as transforms
from .customdataset.ctransforms import *
from .customdataset.dataset import *


def loaddataset(datasetname,experimentparam,batch_size=1,use_cuda=False,worker=1,config_file='defaults/dataconfig_train.json'): 
    print('Loading dataset: ',datasetname)
    
    #load dataset configuration (json)
    data_props = get_data_path(name=datasetname,config_file=config_file)
    for key,value in experimentparam.items():
        data_props[key]=value

    if 'transform_param' in data_props:
        transformstr=data_props['transform_param']
        data_props.pop('transform_param',None)
    else:
        raise 'Please define a default transform \'transform_param\' behavior in '+config_file

    #setup transforms
    transform = eval('transforms.Compose(['+transformstr+'])')

    #dataset 
    datasets = dataset(**data_props,transform_param=transform)

    #loader
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=worker, pin_memory=use_cuda)

    return datasets, loader


def get_data_path(name, config_file='defaults/dataconfig_train.json'):
    data = json.load(open(config_file))
    if name not in data:
        raise 'Dataset '+name+' not found in '+config_file
    return data[name]
