import numpy as np
from dataloaders.imageutl import *
from torch.utils import data
import random


class dataset(data.Dataset):
    def __init__(self, root, ext='jpg', ifolder='image', transform_param=None,burstsize='14'):
        self.root = root #root path
        self.transform_param=transform_param #transforms
        self.dataprov = imageProvide( self.root, fn_image=ifolder,ext=ext) #image provider

    def __len__(self):
        return self.dataprov.num

    def __getitem__(self, index):
        np.random.seed( random.randint(0, 2**32))

        img = self.dataprov.getimage(index)
        if img.ndim==2:
            img=np.repeat(img[:,:,np.newaxis],3,axis=2)

        sample = {'image': img, 'label':np.array([ 0, 1])}

        if self.transform_param is not None:
            sample = self.transform_param(sample)

        return sample
