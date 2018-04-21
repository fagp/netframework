from utils.utils import get_class
import json
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from utils.utils import Decoder
from importlib import import_module


def loaddataset(datasetname,experimentparam,batch_size=1,use_cuda=False,worker=1,config_file='defaults/dataconfig_train.json'): 
    print('Loading dataset: ',datasetname)
    
    #load dataset configuration (json)
    data_props = get_data_path(name=datasetname,config_file=config_file)
    module=data_props['module']
    data_props.pop('module',None)

    for key,value in experimentparam.items():
        data_props[key]=value

    if 'transform_param' in data_props:
        transformstr=data_props['transform_param']
        data_props.pop('transform_param',None)
    else:
        raise 'Please define a default transform \'transform_param\' behavior in '+config_file

    #setup transforms
    tr = import_module( 'dataloaders.'+module+'.ctransforms' )
    transformlist=transformstr.replace(' ','').split('),')
    transformstr=''
    for transf in transformlist:
        transformstr += 'tr.'+transf+'),'
    transformstr=transformstr[:-2]

    transform = eval('transforms.Compose(['+transformstr+'])')

    cdataset=get_class('dataloaders.'+module+'.dataset.cdataset')

    #dataset 
    ddatasets = cdataset(**data_props,transform_param=transform)

    #loader
    tsampler = SubsetRandomSampler(np.random.permutation(len(ddatasets)))
    dloader = DataLoader(ddatasets, batch_size=batch_size, sampler=tsampler, num_workers=worker, pin_memory=use_cuda)

    return ddatasets, dloader


def get_data_path(name, config_file='defaults/dataconfig_train.json'):
    data = json.load(open(config_file),cls=Decoder)
    if name not in data:
        raise 'Dataset '+name+' not found in '+config_file
    return data[name]
