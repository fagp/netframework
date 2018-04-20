from utils.utils import loadnet
import argparse
import os
from torch.autograd import Variable
from utils.utils import warp_Variable
from dataloaders.burstblurdataset.ctransforms import *
import torchvision.transforms as transforms
from torch.utils import data
from scipy.misc import imsave
import torch.nn.functional as F
from dataloaders.imageutl import *
import numpy as np
import ntpath
import torch
from  visdom import Visdom
import time

from dataloaders.burstblurdataset.labelburst import PSNR
from dataloaders.burstblurdataset import render
from dataloaders.burstblurdataset.labelburst import FBA

def switch_images(burst,i,j):
    image_i=burst[3*i:3*i+3:,:,:].clone()
    burst[3*i:3*i+3,:,:]=burst[3*j:3*j+3,:,:].clone()
    burst[3*j:3*j+3,:,:]=image_i.clone()
    return burst


def main():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--experimentpath', nargs='?', type=str, default='../out/blur_TEST', help='Probability map output folder')
    parser.add_argument('--modelpath', nargs='?', type=str, default='../out/UNET1/model/bestmodel.t7', help='Path to pretrained model to be used')
    parser.add_argument('--imsize', nargs='?', type=int, default=200, help='Image resize')
    parser.add_argument('--imagepath', nargs='?', type=str, default='/media/user/DEAE1704AE16D4BD/Documents/FIDEL/viisar_cinmoto-deepblur/db/train/000000363181.jpg', help='Path to test image')
    parser.add_argument('--use_cuda', action='store_true', help='if --use_cuda flag used then computation is done in GPU, otherwise in CPU')
    
    args = parser.parse_args()
    vis = Visdom()
    vis.close(env='blurtest')    
    visimshow = gph.ImageVisdom(env_name='blurtest')
    vistext = gph.TextVisdom(env_name='blurtest')

     # create outputs folder    
    folders={ 1:args.experimentpath}

    for folder, path in folders.items():
        if not os.path.isdir(path):
            os.mkdir(path)

    net,archname=loadnet(args.modelpath)
    if not args.use_cuda:
        net=net.cpu()

    transform_test = transforms.Compose([
        RandomCropImage((args.imsize,args.imsize)),
        ResizeImage(args.imsize),
        ToTensorImage(),
    ])

    iprov=imageProvide('./','','')
    image=np.array(iprov._loadimage( args.imagepath))
    if image.ndim==2:
            image=np.repeat(image[:,:,np.newaxis],3,axis=2)

    pSFsize=64
    maxTotalLength=64
    anxiety=0.05
    numT=300
    texp=0.75
    lmax=100
    burstsize=9

    gen = render.BlurRender(pSFsize=pSFsize,maxTotalLength=maxTotalLength,anxiety=anxiety,numT=numT,texp=texp,lmax=lmax)

    
    burst=np.zeros((image.shape[0],image.shape[1],3*(burstsize+1)))
    metric=np.zeros((burstsize,))
    for i in range(burstsize):
        imblur, _, _ = gen.generatecurve( image )
        burst[:,:,3*i:(3*i)+3]=(imblur*255).astype('uint8')
        metric[i]=PSNR((imblur*255).astype('uint8'),image)
    burst[:,:,3*burstsize:(3*burstsize)+3]=image

    lbl=np.argsort(metric)
    print(lbl)

    burst=transform_test(burst)
    npimage=burst[3*burstsize:(3*burstsize)+3,:,:].numpy().transpose((1,2,0))
    burst=burst[0:3*burstsize,:,:]
    # burst=burst[np.newaxis,:,:,:]

    if args.use_cuda:
        burst = Variable(burst.cuda(),requires_grad=False)
    else:
        burst = Variable(burst,requires_grad=False)

    print('BBsorting')
    for i in range(burstsize):
        visimshow.show('Image'+str(i),burst[3*i:3*i+3,:,:].cpu().numpy())


    oburst=burst.clone()
    t1=time.time()
    for i in range(burstsize-1):
        for j in range(burstsize-1):
            bb=torch.cat( (burst[3*j:3*j+3,:,:],burst[3*(j+1):3*(j+1)+3,:,:]), 0)
            bb=bb[np.newaxis,:,:,:]
            outputs,_ = net(bb)
            
            maxprob,classific=torch.max(F.softmax(outputs,1),1)


            #if classific[0].item()==0:
            #    vistext.show('CL Better','Left')
            #else:
            #    vistext.show('CL Better','Right')
            #visimshow.show('bubble im1',bb[0,0:3,:,:].cpu().numpy())
            #visimshow.show('bubble im2',bb[0,3:6,:,:].cpu().numpy())

            if classific.item()==1:
                burst=switch_images(burst,j,j+1)

            # for k in range(burstsize):
            #     visimshow.show('Image'+str(k),burst[3*k:3*k+3,:,:].cpu().numpy())

    print('Sorting time ',time.time()-t1)
    for i in range(burstsize):
        visimshow.show('Original Image'+str(i),burst[3*i:3*i+3,:,:].cpu().numpy())

    npburst=burst.cpu().numpy().transpose((1,2,0))
    idx=np.zeros((burstsize,))
    idx[0]=1; 
    up=npimage.copy()
    npimage=(npimage*255).astype('uint8')
    

    for i in range(1,burstsize):
        idx[i]=1
        up1,_,_=FBA(npburst,p=11,idx=idx,fftlib='np')
        psmetric=PSNR((up1*255).astype('uint8'),npimage)
        visimshow.show('fb'+str(i)+' psnr '+str(psmetric),up1.transpose((2,0,1)))
        bb=transform_test( np.concatenate(( (up*255).astype(np.uint8), (up1*255).astype(np.uint8) ),axis=2))
    
        bb=bb[np.newaxis,:,:,:]
        if args.use_cuda:
            bb = Variable(bb.cuda(),requires_grad=False)
        else:
            bb = Variable(bb,requires_grad=False)

        outputs,_ = net(bb)
        prob=F.softmax(outputs,1)
        maxprob,classific=torch.max(prob ,1)
        up=up1.copy()

        print(prob[0,0].item())
        if (prob[0,0].item()-0.5)<0.15:
            break



if __name__ == '__main__':
    main()