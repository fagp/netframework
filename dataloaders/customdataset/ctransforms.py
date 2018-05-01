import torch
import numpy as np
from skimage.transform import rotate
from skimage.transform import resize


#########################################################################################################
class ToTensor(object):

    def __call__(self, sample):
        image,label=sample['image'],sample['label']

        image = np.array((image/255.).transpose((2, 0, 1)))

        return {'image': torch.from_numpy(image).float(), 'label': torch.from_numpy(label).long() }
#########################################################################################################
class Rotation(object):
   
    def __init__(self, angle):
            self.angle = angle

    def __call__(self, sample):
        image,label=sample['image'],sample['label']

        angle_rand = np.random.uniform(-self.angle, self.angle)

        image=rotate(image,angle_rand,mode='reflect',preserve_range=True).astype('uint8')
        
        return {'image':image,'label':label}
#########################################################################################################
class RandomFlip(object):
   
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        image,label=sample['image'],sample['label']
       
        if np.random.rand(1) < self.prob:
            image = np.fliplr(image)

        return {'image':image,'label':label}
#########################################################################################################
class RandomCrop(object):
   
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image,label=sample['image'],sample['label']
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        resh=0
        resw=0
        if new_h>=h:
            resh=new_h-h+2
        if new_w>=w:
            resw=new_w-w+2
        
        image=np.lib.pad(image,((resh //2, resh //2), (resw //2, resw //2), (0,0)), 'constant',constant_values=0)
        h, w = image.shape[:2]

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,left: left + new_w,:]

        return {'image':image,'label':label}
#########################################################################################################
class Resize(object):
   
    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, sample):
        image,label=sample['image'],sample['label']

        w = self.output_size
        h = w

        #resize mantaining aspect ratio
        image_x = resize(image.copy(), (h,w),preserve_range=True,mode='reflect', order=1).astype('uint8')

        return {'image':image_x,'label':label}
#########################################################################################################
class ToTensorImage(object):

    def __call__(self, image):

        image = np.array((image/255.).transpose((2, 0, 1)))

        return torch.from_numpy(image).float()
#########################################################################################################
class RandomCropImage(object):
   
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        resh=0
        resw=0
        if new_h>=h:
            resh=new_h-h+2
        if new_w>=w:
            resw=new_w-w+2
        
        image=np.lib.pad(image,((resh //2, resh //2), (resw //2, resw //2), (0,0)), 'constant',constant_values=0)
        h, w = image.shape[:2]

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,left: left + new_w,:]

        return image
#########################################################################################################
class ResizeImage(object):
   
    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, image):

        w = self.output_size
        h = w

        #resize mantaining aspect ratio
        image_x = resize(image.copy(), (h,w),preserve_range=True,mode='reflect', order=1).astype('uint8')

        return image_x