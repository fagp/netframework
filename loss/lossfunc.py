import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        pass

    def forward(self,input,target):

        classific=torch.argmax(F.softmax(input,1) ,1)
        res=(classific).eq(target).sum(0).float() / float(target.size(0))

        return res
