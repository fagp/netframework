import torch
import random
import numpy as np
from torch.utils import data
from torch.autograd import Variable

from netframework.dataloaders.imageutl import imageProvide

class cdataset(data.Dataset):
    def __init__(self, root, ext='jpg', ifolder='image', transform_param=None,datasize=100):
        self.root = root #root path
        self.datasize=datasize
        self.transform_param=transform_param #transforms
        self.dataprov = imageProvide( self.root, fn_image=ifolder,ext=ext) #image provider

    def __len__(self):
        return self.datasize#self.dataprov.num

    def __getitem__(self, index):
        np.random.seed( random.randint(0, 2**32))

        img = self.dataprov.getimage(index)
        if img.ndim==2:
            img=np.repeat(img[:,:,np.newaxis],3,axis=2)

        rnd=np.random.randint(0,2)
        sample = {'image': img, 'label':np.array([ rnd, 1-rnd])}

        if self.transform_param is not None:
            sample = self.transform_param(sample)

        return sample

def warp_Variable(sample,device):
    images, labels = sample['image'], sample['label']
    images=images.to(device)
    labels=labels.to(device)

    sample = {'image': images, 'label':labels.squeeze(1)}
    return sample
