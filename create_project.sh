#!/bin/bash

mkdir ../dataloaders
mkdir ../dataloaders/customdataset
touch ../dataloaders/__init__.py
touch ../dataloaders/customdataset/__init__.py
touch ../dataloaders/customdataset/dataset.py
touch ../dataloaders/customdataset/ctransforms.py

mkdir ../defaults
touch ../defaults/dataconfig_test.json
touch ../defaults/dataconfig_train.json
touch ../defaults/loss_definition.json
touch ../defaults/metrics.json
touch ../defaults/modelconfig.json

mkdir ../loss
touch ../loss/__init__.py
touch ../loss/lossfunc.py

mkdir ../models
mkdir ../models/arch
touch ../models/__init__.py
touch ../models/arch/__init__.py

mkdir ../netutil
touch ../netutil/__init__.py
touch ../netutil/customnet.py

echo "import os"  >> ../dataloaders/customdataset/ctransforms.py
echo "import torch" >> ../dataloaders/customdataset/ctransforms.py
echo "import torchvision" >> ../dataloaders/customdataset/ctransforms.py
echo "import math" >> ../dataloaders/customdataset/ctransforms.py
echo "import numpy as np" >> ../dataloaders/customdataset/ctransforms.py
echo "import scipy.misc as m" >> ../dataloaders/customdataset/ctransforms.py
echo "from scipy import ndimage" >> ../dataloaders/customdataset/ctransforms.py
echo "import skimage.color as skcolor" >> ../dataloaders/customdataset/ctransforms.py
echo "import skimage.util as skutl" >> ../dataloaders/customdataset/ctransforms.py
echo "from scipy.interpolate import griddata" >> ../dataloaders/customdataset/ctransforms.py
echo "from skimage.transform import rotate" >> ../dataloaders/customdataset/ctransforms.py
echo "from skimage.transform import resize" >> ../dataloaders/customdataset/ctransforms.py
echo "from torch.utils import data" >> ../dataloaders/customdataset/ctransforms.py
echo "import time" >> ../dataloaders/customdataset/ctransforms.py
echo "import itertools" >> ../dataloaders/customdataset/ctransforms.py
echo "from torch.autograd import Variable" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "#########################################################################################################" >> ../dataloaders/customdataset/ctransforms.py
echo "class ToTensor(object):" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "    def __call__(self, sample):" >> ../dataloaders/customdataset/ctransforms.py
echo "        image=sample['image']" >> ../dataloaders/customdataset/ctransforms.py
echo "        #add anything you need" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "        image = np.array((image/255.).transpose((2, 0, 1)))" >> ../dataloaders/customdataset/ctransforms.py
echo "      " >> ../dataloaders/customdataset/ctransforms.py
echo "        return {'image': torch.from_numpy(image).float()} #add anything you need" >> ../dataloaders/customdataset/ctransforms.py
echo "#########################################################################################################" >> ../dataloaders/customdataset/ctransforms.py
echo "class RandomCrop(object):" >> ../dataloaders/customdataset/ctransforms.py
echo "    '''" >> ../dataloaders/customdataset/ctransforms.py
echo "    Example of random crop transformation" >> ../dataloaders/customdataset/ctransforms.py
echo "    '''" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "    def __init__(self, output_size):" >> ../dataloaders/customdataset/ctransforms.py
echo "        assert isinstance(output_size, (int, tuple))" >> ../dataloaders/customdataset/ctransforms.py
echo "        if isinstance(output_size, int):" >> ../dataloaders/customdataset/ctransforms.py
echo "            self.output_size = (output_size, output_size)" >> ../dataloaders/customdataset/ctransforms.py
echo "        else:" >> ../dataloaders/customdataset/ctransforms.py
echo "            assert len(output_size) == 2" >> ../dataloaders/customdataset/ctransforms.py
echo "            self.output_size = output_size" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "    def __call__(self, sample):" >> ../dataloaders/customdataset/ctransforms.py
echo "        image=sample['image']" >> ../dataloaders/customdataset/ctransforms.py
echo "        #add anything you need" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "        h, w = image.shape[:2]" >> ../dataloaders/customdataset/ctransforms.py
echo "        new_h, new_w = self.output_size" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "        top = np.random.randint(0, h - new_h)" >> ../dataloaders/customdataset/ctransforms.py
echo "        left = np.random.randint(0, w - new_w)" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "        image = image[top: top + new_h,left: left + new_w,:]" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py
echo "        return {'image':image}#add anything you need" >> ../dataloaders/customdataset/ctransforms.py
echo "" >> ../dataloaders/customdataset/ctransforms.py

echo "import os" >> ../dataloaders/customdataset/dataset.py
echo "import torch" >> ../dataloaders/customdataset/dataset.py
echo "import random" >> ../dataloaders/customdataset/dataset.py
echo "import torchvision" >> ../dataloaders/customdataset/dataset.py
echo "import numpy as np" >> ../dataloaders/customdataset/dataset.py
echo "import scipy.misc as m" >> ../dataloaders/customdataset/dataset.py
echo "from skimage import filters" >> ../dataloaders/customdataset/dataset.py
echo "from torch.utils import data" >> ../dataloaders/customdataset/dataset.py
echo "import skimage.util as skutl" >> ../dataloaders/customdataset/dataset.py
echo "import skimage.color as skcolor" >> ../dataloaders/customdataset/dataset.py
echo "from torch.autograd import Variable" >> ../dataloaders/customdataset/dataset.py
echo "" >> ../dataloaders/customdataset/dataset.py
echo "#for load images. might be used" >> ../dataloaders/customdataset/dataset.py
echo "from netframework.dataloaders.imageutl import *" >> ../dataloaders/customdataset/dataset.py
echo "" >> ../dataloaders/customdataset/dataset.py
echo "" >> ../dataloaders/customdataset/dataset.py
echo "class cdataset(data.Dataset):" >> ../dataloaders/customdataset/dataset.py
echo "    def __init__(self, root, ext='jpg', ifolder='image',transform_param=None):" >> ../dataloaders/customdataset/dataset.py
echo "	    #do some initialization" >> ../dataloaders/customdataset/dataset.py
echo "        self.transform_param=transform_param" >> ../dataloaders/customdataset/dataset.py
echo "" >> ../dataloaders/customdataset/dataset.py
echo "    def __len__(self):" >> ../dataloaders/customdataset/dataset.py
echo "        return 1#dataset lenght" >> ../dataloaders/customdataset/dataset.py
echo "" >> ../dataloaders/customdataset/dataset.py
echo "    def __getitem__(self, index):" >> ../dataloaders/customdataset/dataset.py
echo "        np.random.seed( random.randint(0, 2**32))" >> ../dataloaders/customdataset/dataset.py
echo "        " >> ../dataloaders/customdataset/dataset.py
echo "        #do something" >> ../dataloaders/customdataset/dataset.py
echo "        image=np.zeros((10,10,3))" >> ../dataloaders/customdataset/dataset.py
echo "        " >> ../dataloaders/customdataset/dataset.py
echo "        #dictionary with image key." >> ../dataloaders/customdataset/dataset.py
echo "        sample = {'image': image} #add what you need" >> ../dataloaders/customdataset/dataset.py
echo "" >> ../dataloaders/customdataset/dataset.py
echo "        if self.transform_param is not None:" >> ../dataloaders/customdataset/dataset.py
echo "            sample = self.transform_param(sample)" >> ../dataloaders/customdataset/dataset.py
echo "" >> ../dataloaders/customdataset/dataset.py
echo "        return sample" >> ../dataloaders/customdataset/dataset.py
echo "" >> ../dataloaders/customdataset/dataset.py
echo "def warp_Variable(sample,device):" >> ../dataloaders/customdataset/dataset.py
echo "    images = sample['image']" >> ../dataloaders/customdataset/dataset.py
echo "    images=images.to(device)" >> ../dataloaders/customdataset/dataset.py
echo "    #add other elements of the dictionary" >> ../dataloaders/customdataset/dataset.py
echo "    " >> ../dataloaders/customdataset/dataset.py
echo "    sample = {'image': images,'label': torch.tensor([0]).to(device)} #wrap it again. add what you need" >> ../dataloaders/customdataset/dataset.py
echo "    return sample" >> ../dataloaders/customdataset/dataset.py

echo '{' >> ../defaults/dataconfig_test.json
echo '  "dataset1":' >> ../defaults/dataconfig_test.json
echo '  {' >> ../defaults/dataconfig_test.json
echo '    "root": "data/dataset1",' >> ../defaults/dataconfig_test.json
echo '    "ifolder": "test",' >> ../defaults/dataconfig_test.json
echo '    "ext": "png",' >> ../defaults/dataconfig_test.json
echo '    "transform_param": "RandomCrop(3),ToTensor()",' >> ../defaults/dataconfig_test.json
echo '    "module":"dataloaders.customdataset"' >> ../defaults/dataconfig_test.json
echo '  }' >> ../defaults/dataconfig_test.json
echo '}' >> ../defaults/dataconfig_test.json

echo '{' >> ../defaults/dataconfig_train.json
echo '  "dataset1":' >> ../defaults/dataconfig_train.json
echo '  {' >> ../defaults/dataconfig_train.json
echo '    "root": "data/dataset1",' >> ../defaults/dataconfig_train.json
echo '    "ifolder": "train",' >> ../defaults/dataconfig_train.json
echo '    "ext": "png",' >> ../defaults/dataconfig_train.json
echo '    "transform_param": "RandomCrop(3),ToTensor()",' >> ../defaults/dataconfig_train.json
echo '    "module":"dataloaders.customdataset"' >> ../defaults/dataconfig_train.json
echo '  }' >> ../defaults/dataconfig_train.json
echo '}' >> ../defaults/dataconfig_train.json

echo '{' >> ../defaults/loss_definition.json
echo '  "ce":' >> ../defaults/loss_definition.json
echo '  {' >> ../defaults/loss_definition.json
echo '    "criterion":"CrossEntropyLoss",' >> ../defaults/loss_definition.json
echo "    \"criterionparam\":\"{'input':outputs,'target':sample['label']}\"" >> ../defaults/loss_definition.json
echo '  },' >> ../defaults/loss_definition.json
echo '  "accuracy":' >> ../defaults/loss_definition.json
echo '  {' >> ../defaults/loss_definition.json
echo '    "criterion":"Accuracy",' >> ../defaults/loss_definition.json
echo "    \"criterionparam\":\"{'input':outputs,'target':sample['label']}\"," >> ../defaults/loss_definition.json
echo '    "module":"loss.lossfunc"' >> ../defaults/loss_definition.json
echo '  }' >> ../defaults/loss_definition.json
echo '}' >> ../defaults/loss_definition.json

echo '{' >> ../defaults/metrics.json
echo '  "accuracy":' >> ../defaults/metrics.json
echo '  {' >> ../defaults/metrics.json
echo '    "metric":"accuracy",' >> ../defaults/metrics.json
echo '    "param":{}' >> ../defaults/metrics.json
echo '  }' >> ../defaults/metrics.json
echo '}' >> ../defaults/metrics.json

echo '{' >> ../defaults/modelconfig.json
echo '  "mynet":' >> ../defaults/modelconfig.json
echo '  {' >> ../defaults/modelconfig.json
echo '    "arch":"MyNet",' >> ../defaults/modelconfig.json
echo '    "module": "models.arch.net1",' >> ../defaults/modelconfig.json
echo '    "n_classes":"2"' >> ../defaults/modelconfig.json
echo '  }' >> ../defaults/modelconfig.json
echo '}' >> ../defaults/modelconfig.json

echo "import torch" >> ../loss/lossfunc.py
echo "import numpy as np" >> ../loss/lossfunc.py
echo "import torch.nn as nn" >> ../loss/lossfunc.py
echo "import torch.nn.functional as F" >> ../loss/lossfunc.py
echo "" >> ../loss/lossfunc.py
echo "#########################################################################################################" >> ../loss/lossfunc.py
echo "class Accuracy(nn.Module):" >> ../loss/lossfunc.py
echo "    def __init__(self):" >> ../loss/lossfunc.py
echo "        super(Accuracy, self).__init__()" >> ../loss/lossfunc.py
echo "        pass" >> ../loss/lossfunc.py
echo "" >> ../loss/lossfunc.py
echo "    def forward(self,input,target):" >> ../loss/lossfunc.py
echo "	#implement your accuracy function here" >> ../loss/lossfunc.py
echo "        " >> ../loss/lossfunc.py
echo "        return torch.ones(1)*100" >> ../loss/lossfunc.py
echo "" >> ../loss/lossfunc.py

echo "import torch" >> ../models/arch/net1.py
echo "import torch.nn as nn" >> ../models/arch/net1.py
echo "" >> ../models/arch/net1.py
echo "#########################################################################################################" >> ../models/arch/net1.py
echo "class MyNet(nn.Module):" >> ../models/arch/net1.py
echo "" >> ../models/arch/net1.py
echo "    def __init__(self, n_classes=2):" >> ../models/arch/net1.py
echo "        super(MyNet, self).__init__()" >> ../models/arch/net1.py
echo "        #define architecture here" >> ../models/arch/net1.py
echo "        self.conv1 = nn.Sequential(nn.Conv2d(3, 2, 3, 1, 0)," >> ../models/arch/net1.py
echo "                                       nn.ReLU(),)" >> ../models/arch/net1.py
echo "" >> ../models/arch/net1.py
echo "    def forward(self, inputs):" >> ../models/arch/net1.py
echo "        #implement forward" >> ../models/arch/net1.py
echo "        out=self.conv1(inputs)" >> ../models/arch/net1.py
echo "" >> ../models/arch/net1.py
echo "        return out.squeeze(2).squeeze(2)" >> ../models/arch/net1.py
echo "" >> ../models/arch/net1.py

echo "import torch" >> ../netutil/customnet.py
echo "from netframework.netutil.NetFramework import NetFramework" >> ../netutil/customnet.py
echo "import torch.nn.functional as F" >> ../netutil/customnet.py
echo "from scipy.misc import imsave" >> ../netutil/customnet.py
echo "import os" >> ../netutil/customnet.py
echo "import numpy as np" >> ../netutil/customnet.py
echo "from scipy.io import savemat, loadmat" >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "class CustomNet(NetFramework):" >> ../netutil/customnet.py
echo "    def __init__(self,default_path):" >> ../netutil/customnet.py
echo "        NetFramework.__init__(self,default_path)" >> ../netutil/customnet.py
echo "        pass" >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "    def valid_visualization(self,current_epoch,index=0,save=False):  " >> ../netutil/customnet.py
echo "        index=0" >> ../netutil/customnet.py
echo "        " >> ../netutil/customnet.py
echo "        with torch.no_grad():" >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "            sample=self.testdataset[ index ]" >> ../netutil/customnet.py
echo "            sample['image'].unsqueeze_(0)" >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "            sample=self.warp_var_mod.warp_Variable(sample,self.device)" >> ../netutil/customnet.py
echo "            images=sample['image']" >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "            outputs = self.net(images)       " >> ../netutil/customnet.py
echo "            prob=F.softmax(outputs,dim=1)" >> ../netutil/customnet.py
echo "            prob=prob.detach()[0]" >> ../netutil/customnet.py
echo "            _,maxprob=torch.max(prob,0)       " >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "            if self.visdom==True:" >> ../netutil/customnet.py
echo "                self.visheatmap.show('Image',images.detach().cpu().numpy()[0][0],colormap='Greys',scale=0.5)" >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "            if save==True:" >> ../netutil/customnet.py
echo "                if prob.size(0)>2:" >> ../netutil/customnet.py
echo "                    imsave(os.path.join(self.folders['images_path'],'image-{:d}-{:03d}'.format(index+1,current_epoch) +'.png'), prob.cpu().numpy().transpose((1,2,0))*255)" >> ../netutil/customnet.py
echo "                else:" >> ../netutil/customnet.py
echo "                    imsave(os.path.join(self.folders['images_path'],'image-{:d}-{:03d}'.format(index+1,current_epoch) +'.png'), prob[1].cpu().numpy()*255)" >> ../netutil/customnet.py
echo "" >> ../netutil/customnet.py
echo "        return 1" >> ../netutil/customnet.py

echo "import time" >> ../train.py
echo "from netutil.customnet import CustomNet" >> ../train.py
echo "" >> ../train.py
echo "def main():" >> ../train.py
echo "    net = CustomNet('defaults')" >> ../train.py
echo "    start=time.time()" >> ../train.py
echo "    net.do_train()" >> ../train.py
echo "    print('Total Training Time {:.3f}'.format(time.time()-start))" >> ../train.py
echo "" >> ../train.py
echo "if __name__ == '__main__':" >> ../train.py
echo "    main()" >> ../train.py
