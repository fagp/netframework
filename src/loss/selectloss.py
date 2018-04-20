from loss.lossfunc import *
import torch.nn as nn

def selectloss(lossname,parameter=None,use_cuda=False):
    print('Selecting loss function: ',lossname)
    if lossname=='CE':
        criterion = nn.CrossEntropyLoss(size_average=True)
        criterioneval="self.criterion(outputs,labels[:,0])"
    elif lossname=='MSE':
        criterion = nn.MSELoss(size_average=True)
        criterioneval="self.criterion(outputs[:,0],labels[:,0])"
    elif lossname=='L1':
        criterion = nn.L1Loss(size_average=True)
        criterioneval="self.criterion(outputs[:,0],labels[:,0])"
    else:
        raise 'Loss {} not available'.format(lossname)

    if torch.cuda.is_available() and use_cuda:
        criterion=criterion.cuda()

    return criterion,criterioneval
