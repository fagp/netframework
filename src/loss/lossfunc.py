import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import graphics as gph

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        pass

    def forward(self,input,target,weights=None):
        n, c, h, w = input.size()
        nt,ht,wt= target.size()

        offset_w,offset_h=(wt-w) // 2 ,(ht-h) // 2

        if offset_w>0 and offset_h>0:
            target=target[:,offset_h:-offset_h,offset_w:-offset_w]

        prob=F.softmax(input,dim=1)
        prob=prob.data
        _,maxprob=torch.max(prob,1)
        
        correct=0
        for cl in range(c):
            correct+= (((maxprob.eq(cl) + target.data.eq(cl)).eq(2)).view(-1).float().sum(0) +1) / (target.data.eq(cl).view(-1).float().sum(0) + 1)

        correct=correct/c
        res=correct.mul_(100.0)

        return res
