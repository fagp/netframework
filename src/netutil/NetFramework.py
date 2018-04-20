import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from scipy.misc import imsave
import gc
import time

import os
from dataloaders.loaddataset import loaddataset
from utils import graphics as gph
from utils.utils import *
from models.loadmodel import loadmodel
from optimizers.selectopt import selectoptimizer
from optimizers.selectschedule import selectschedule
from loss.selectloss import selectloss
from loss.lossfunc import Accuracy
from  visdom import Visdom
import random

class NetFramework():
    def __init__(self, args):

        # Parse use cuda
        self.use_cuda, self.use_parallel, self.ngpu = parse_cuda(args)
        torch.cuda.set_device(self.ngpu)

        # Visdom visualization
        self.visdom=args.visdom
        if self.visdom==True:
            self.vis = Visdom()
            self.vis.close(env=args.experiment)
            self.visplotter = gph.VisdomLinePlotter(env_name=args.experiment)
            self.visheatmap = gph.HeatMapVisdom(env_name=args.experiment)
            self.visimshow  = gph.ImageVisdom(env_name=args.experiment)
            self.vistext    = gph.TextVisdom(env_name=args.experiment)

        # Showing results rate
        self.print_rate = args.print_rate
        self.show_rate = args.show_rate
        self.save_rate = args.save_rate
        
        self.init_epoch=0
        self.current_epoch=0
        self.epochs=args.epochs
        self.folders=args.folders
        self.bestmetric=0

        # Load datasets
        _,self.train_loader = loaddataset(datasetname=args.dataset,
                                        experimentparam=args.datasetparam,
                                        batch_size=args.batch_size,
                                        use_cuda=self.use_cuda,
                                        worker=args.train_worker,
                                        config_file='defaults/dataconfig_train.json')
        
        _,self.test_loader = loaddataset(datasetname=args.dataset,
                                        experimentparam=args.datasetparam,
                                        batch_size=args.batch_size,
                                        use_cuda=self.use_cuda,
                                        worker=args.test_worker,
                                        config_file='defaults/dataconfig_test.json')

        # Setup model
        self.net, self.arch = loadmodel(modelname=args.model,
                                        experimentparams=args.modelparam,
                                        use_cuda=self.use_cuda,
                                        use_parallel=self.use_parallel,
                                        config_file='defaults/modelconfig.json')

        # Setup Optimizer
        self.optimizer = selectoptimizer(args.optimizer,self.net,args.optimizerparam)
        
        if args.resume:
            self.resume()
            
        # Setup Learning Rate Scheduling
        self.scheduler = selectschedule(args.lrschedule, self.optimizer)

        # Setup Loss criterion
        self.criterion, self.criterioneval = selectloss(args.loss,args.lossparam)
        # Others evaluation metrics
        self.accuracy = Accuracy()

        self.trlosses = AverageMeter()
        self.tracc = AverageMeter()
        self.tslosses = AverageMeter()
        self.tsacc = AverageMeter()


    def resume(self):
        if os.path.isdir(self.folders['model_path']):
            files = [ f for f in sorted(os.listdir(self.folders['model_path'])) if (f.find('epoch')!=-1 and f.find('model.t7')!=-1) ]
            if files:
                self.init_epoch = max(int(files[5:files.find('model.t7')]))+1
                self.loadmodel(os.path.join(self.folders['model_path'], 'epoch'+str(args.initial_epoch-1)+'model.t7' ))


    def do_train(self):
        for current_epoch in range(self.init_epoch,self.epochs):
            print('epoch ',current_epoch)
            self.current_epoch=current_epoch
            # Forward over validation set
            avgloss, avgacc=self.validation(current_epoch)
            self.scheduler.step(avgloss, current_epoch)

            # Save probability map of image "3" after self.save_rate epochs
            # print('| Plotting test image:')
            save_image = True if self.save_rate!=0 and (current_epoch % self.save_rate)==0 else False
            self.test(current_epoch,3,save_image)

            # If obtained validation accuracy improvement save network in model/bestmodel.t7
            if self.bestacc<avgacc:
                print('Validation accuracy improvement ({}) in epoch {} \n'.format(avgacc,current_epoch))
                self.bestacc=avgacc
                to_save= self.net.module if self.use_parallel else self.net
                savemodel(to_save, os.path.join(self.experimentpath,'model/bestmodel.t7'),self.arch)

            
            # Save netowrk after self.save_rate epochs
            if save_image:
                to_save= self.net.module if self.use_parallel else self.net
                savemodel(to_save, os.path.join(self.experimentpath,'model/epoch{}model.t7'.format(current_epoch) ),self.arch)
                saveoptimizer(self.optimizer,os.path.join(self.experimentpath,'model/epoch{}optimizer.t7'.format(current_epoch) ) )
                metrics_dict={'train_loss':self.trlosses,'train_accuracy':self.tracc, 'validation_loss':self.tslosses, 'validation_accuracy':self.tsacc}

                for tag, value in metrics_dict.items():
                    np.savetxt(self.experimentpath+'/'+tag+'.txt', np.array(value.array) , delimiter=',', fmt='%3.6f') 

            # Forward and backward over training set
            self.train(current_epoch)
        
        # Save last model netowrk
        to_save= self.net.module if self.use_parallel else self.net
        savemodel(to_save, os.path.join(self.experimentpath,'model/lastmodel.t7'),self.arch)

    ## Train function
    def train(self,current_epoch):
        print('Training')
        data_time = AverageMeter()
        batch_time = AverageMeter()   
        losses = AverageMeter()
        acc = AverageMeter()     

        self.net.train()   

        end = time.time()
        total_train=len(self.train_loader)
        for i, sample in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            iteration=float(i)/len(self.train_loader)+current_epoch

            images,labels=warp_Variable(sample,self.use_cuda,Volatile=False,Grad=True)
            labels=labels.squeeze(1)
            self.optimizer.zero_grad()
            outputs,_ = self.net(images)
            loss=eval(self.criterioneval)
            loss.backward()
            self.optimizer.step()

            _,classific=torch.max(F.softmax(outputs,1) ,1)
            acc_metric=(classific).eq(labels[:,0]).sum(0).item() / float(labels.size(0))

            self.trlosses.update(loss.item(),images.size(0))
            losses.update(loss.item(),images.size(0))
            self.tracc.update(acc_metric,images.size(0))
            acc.update(acc_metric,images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i % self.print_rate)==0:
                strinfo  = '| Epoch: [{0}][{1}/{2}]\t\t'                
                strinfo += 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'     
                strinfo += 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                strinfo += 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                strinfo += 'Acc {acc.val:.3f} ({acc.avg:.3f})'   

                print(
                        strinfo.format(
                            current_epoch, i, len(self.train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=self.trlosses,
                            acc=self.tracc
                            )                
                        )

            if self.visdom==True and (  ((i+1) % self.show_rate)==0 or ((i+1) % total_train)==0 ):
                info = {
                        'loss':self.trlosses, 
                        'accuracy':self.tracc
                        }

                for tag, value in info.items():
                    self.visplotter.show(tag, 'train_mean', iteration, value.avg )
                
                info = {
                        'loss':losses, 
                        'accuracy':acc
                        }

                for tag, value in info.items():
                    self.visplotter.show(tag, 'train', iteration, value.avg )

            del outputs, loss, acc_metric


    def validation(self,current_epoch):   
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter() 
        
        self.net.eval()   

        end = time.time()
        total_valid=len(self.valid_loader)
        for i, sample in enumerate(self.valid_loader):
            data_time.update(time.time() - end)
            iteration=float(i)/len(self.valid_loader)+current_epoch-1

            images,labels=warp_Variable(sample,self.use_cuda,Volatile=True,Grad=False)
            labels=labels.squeeze(1)
            outputs,_ = self.net(images)
            loss=eval(self.criterioneval)
            _,classific=torch.max(F.softmax(outputs,1) ,1)
            acc_metric=(classific).eq(labels[:,0]).sum(0).item() / float(labels.size(0))

            self.tslosses.update(loss.item(),images.size(0))
            losses.update(loss.item(),images.size(0))
            self.tsacc.update(acc_metric,images.size(0))
            acc.update(acc_metric,images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i%self.print_rate==0:
                strinfo  = '| Validation: [{0}][{1}/{2}]\t\t'                
                strinfo += 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'     
                strinfo += 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                strinfo += 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                strinfo += 'Acc {acc.val:.3f} ({acc.avg:.3f})'    

                print(
                        strinfo.format(
                            current_epoch, i, len(self.valid_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=self.tslosses,
                            acc=self.tsacc
                            )                
                        )

            if self.visdom==True and current_epoch!=self.init_epoch and (  ((i+1) % self.show_rate)==0 or ((i+1) % total_valid)==0 ):
                info = {
                        'loss':self.tslosses, 
                        'accuracy':self.tsacc
                        }

                for tag, value in info.items():
                    self.visplotter.show(tag, 'validation_mean', iteration, value.avg)

                info = {
                        'loss':losses, 
                        'accuracy':acc
                        }

                for tag, value in info.items():
                    self.visplotter.show(tag, 'validation', iteration, value.avg)

            del outputs, loss, acc_metric

        return losses.avg, acc.avg

    def test(self,current_epoch,index=0,save=False):   
        self.net.eval()   

        #i, sample =self.testimages
        self.testimages=self.test_loader[ random.randint(0,len(self.test_loader)) ]
        self.testimages['image'].unsqueeze_(0)
        self.testimages['label'].unsqueeze_(0)
        sample =self.testimages
        images,labels=warp_Variable(sample,self.use_cuda,Grad=False)
        outputs,_ = self.net(images)
        # loss=eval(self.criterioneval)
        _,classific=torch.max(F.softmax(outputs,1) ,1)
        # acc_metric=(classific).eq(labels[:,0]).sum(0).item() / float(labels.size(0))
        
        burst=images.data[0].cpu().numpy()#.transpose((1,2,0))
        if self.visdom==True:
            # self.visplotter.show('loss', 'test', current_epoch, loss.item(),color=np.array([[255,0,0]]))
            # self.visplotter.show('accuracy', 'test', current_epoch, acc_metric,color=np.array([[255,0,0]]))
            self.visimshow.show('Image1',burst[0:3,:,:])
            self.visimshow.show('Image2',burst[3:6,:,:])
            if labels[0,0].item()==0:
                self.vistext.show('GT Better','Left')
            else:
                self.vistext.show('GT Better','Right')
            if classific[0].item()==0:
                self.vistext.show('CL Better','Left')
            else:
                self.vistext.show('CL Better','Right')

        #del outputs
        return 1


    def savemodel(self,modelpath):
        print('Saving..')
        state = {
                'epoch': self.current_epoch,
                'arch':  self.arch,
                'net':   self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'bestmetric': self.bestmetric
            }
        torch.save(state, modelpath)
    
    def loadmodel(self,modelpath):
        if os.path.isfile(modelpath):
            checkpoint = torch.load(modelpath)
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_epoch=checkpoint['epoch']
            self.arch=checkpoint['arch']
            self.bestmetric=checkpoint['bestmetric']
        else:
            raise 'Model not found'