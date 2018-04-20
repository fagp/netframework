from torch import nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import torch
import os

def parse_cuda(args):
    if args.use_cuda>-1:
        print('Using CUDA')
        use_cuda=True
        ngpu=args.use_cuda
        use_parallel=args.parallel
    else:
        print('Using CPU')
        ngpu=-1
        use_cuda=False
        use_parallel=False
    
    return use_cuda, use_parallel, ngpu

def warp_Variable(sample,use_cuda=False,Volatile=False,Grad=True):
    images, labels = sample['image'], sample['label']

    if torch.cuda.is_available() and use_cuda:
        images = Variable(images.cuda(),requires_grad=Grad) #,volatile=Volatile,requires_grad=Grad)
        labels = Variable(labels.cuda(),requires_grad=False)#,volatile=Volatile,requires_grad=False)
    else:
        images = Variable(images,requires_grad=Grad)#,volatile=Volatile,requires_grad=Grad)
        labels = Variable(labels,requires_grad=False)#,volatile=Volatile,requires_grad=False)

    return images,labels

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()        
    def reset(self):
        self.array = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.array= self.array + [val]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count