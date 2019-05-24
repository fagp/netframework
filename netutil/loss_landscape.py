# import gc
import os
# import sys
import json
import time
# import torch
# import random
import signal
import argparse
# import numpy as np
# import torch.nn as nn
from  visdom import Visdom
from ..utils.utils import *
# from scipy.misc import imsave
import torch.nn.functional as F
from ..utils.utils import Decoder
from ..utils.landscape_utils import *
# import torchvision.models as models
# from torch.autograd import Variable
from importlib import import_module
from ..utils import graphics as gph
import torch.backends.cudnn as cudnn
from ..models.loadmodel import loadmodel,init_params
from ..loss.selectloss import selectloss
from ..loss.selectloss import get_metric_path
from ..dataloaders.loaddataset import loaddataset
from ..optimizers.selectopt import selectoptimizer
from ..optimizers.selectschedule import selectschedule
import warnings
import random
import copy
import scipy.io

warnings.filterwarnings("ignore")

class LandscapeVis():
    def __init__(self, defaults_path):

        parser = argparse.ArgumentParser(description='Landscape arguments description')
        parser.add_argument('--experiment', nargs='?', type=str, default='experiment', help='Experiment name')
        parser.add_argument('--surface_name', nargs='?', type=str, default='surface', help='Output surface file name')

        parser.add_argument('--visdom', action='store_true', help='If included shows visdom visulaization')
        parser.add_argument('--show_rate', nargs='?',type=int, default=4, help='Visdom show after num of iterations (used with --visdom)')
        parser.add_argument('--print_rate', nargs='?',type=int, default=4, help='Print after num of iterations')
        parser.add_argument('--save_rate', nargs='?',type=int, default=10, help='Save after num of iterations (if --save_rate=0 then no save is done during training)')

        parser.add_argument('--use_cuda', nargs='?',type=int, default=0, help='GPU device (if --use_cuda=-1 then CPU used)')
        parser.add_argument('--use_dataloader', nargs='?',type=int, default=0, help='Use dataloaders instead dataset iteration')
        parser.add_argument('--parallel', action='store_true', help='Use multiples GPU (used only if --use_cuda>-1)')
        parser.add_argument('--epoch', nargs='?', type=int, default=-1, help='Epoch number')
        parser.add_argument('--worker', nargs='?', type=int, default=0, help='Number of testing workers')
        
        parser.add_argument('--loss', nargs='?', type=str, default='', help='Loss function to use')
        parser.add_argument('--lossparam', type=str, default='{}', help='Loss function parameters')
        
        parser.add_argument('--dataset', nargs='?', type=str, default='', help='Dataset key specified in dataconfig_*.json')
        parser.add_argument('--datasetparam', type=str, default='{}', help='Experiment dataset parameters')
        parser.add_argument('--traindata', action='store_true', help='If included load train dataset, else load dev dataset')
        
        parser.add_argument('--batch_size', nargs='?', type=int, default=-1, help='Minibatch size')
        parser.add_argument('--batch_acc', nargs='?', type=int, default=1, help='Minibatch accumulation')

        parser.add_argument('--seed', nargs='?',type=int, default=123, help='Random seed (for reproducibility)')
        
        parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
        parser.add_argument('--y', default='-1:1:51', help='A string with format ymin:ymax:ynum')
        parser.add_argument('--xnorm', default='filter', help='direction normalization: filter | layer | weight')
        parser.add_argument('--ynorm', default='filter', help='direction normalization: filter | layer | weight')
        parser.add_argument('--xignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
        parser.add_argument('--yignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
        parser.add_argument('--plane', nargs='?', type=str, default='', help='Path to folder containing plane files')

        args = parser.parse_args()

        if args.seed!=-1:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

        # create outputs folders
        root='../out'
        experimentpath=(os.path.join(root,args.experiment))
        if args.plane=='':
            args.plane=os.path.join(experimentpath,'surface')
        folders={ 'root_path':root, 'experiment_path':experimentpath, 'model_path':os.path.join(experimentpath,'model'), 'images_path':os.path.join(experimentpath,'images'),'surface_path':os.path.join(experimentpath,'surface'),
        'plane_path_x': os.path.join(args.plane,'x.t7'),'plane_path_y': os.path.join(args.plane,'y.t7') }

        
        if not os.path.isdir(folders['model_path']):
            raise Exception('Experiment {} not found'.format(path))

        if not os.path.isdir(folders['surface_path']):
            try:
                os.mkdir(folders['surface_path'])  
            except:
                pass

        expargs=json.load(open(os.path.join(experimentpath,'args.json')))
        args.folders=folders

        args.lossparam=json.loads(args.lossparam.replace("'","\""),cls=Decoder)
        args.datasetparam=json.loads(args.datasetparam.replace("'","\""),cls=Decoder)
        expargs['lossparam']=json.loads(expargs['lossparam'].replace("'","\""),cls=Decoder)
        expargs['datasetparam']=json.loads(expargs['datasetparam'].replace("'","\""),cls=Decoder)
        expargs['modelparam']=json.loads(expargs['modelparam'].replace("'","\""),cls=Decoder)

        # Parse use cuda
        self.device, self.use_parallel = parse_cuda(args)
        torch.cuda.set_device(args.use_cuda)

        # Visdom visualization
        self.visdom=args.visdom
        if self.visdom==True:
            self.vis = Visdom(use_incoming_socket=False)
            self.vis.close(env='landscape_'+args.experiment+'_'+args.surface_name)
            self.visplotter = gph.VisdomLinePlotter(self.vis, env_name='landscape_'+args.experiment+'_'+args.surface_name)
            self.visheatmap = gph.HeatMapVisdom(self.vis, env_name='landscape_'+args.experiment+'_'+args.surface_name)
            self.visimshow  = gph.ImageVisdom(self.vis, env_name='landscape_'+args.experiment+'_'+args.surface_name)
            self.vistext    = gph.TextVisdom(self.vis, env_name='landscape_'+args.experiment+'_'+args.surface_name)
            self.vissurf    = gph.VisdomSurface(self.vis, env_name='landscape_'+args.experiment+'_'+args.surface_name)

        try:
            args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
            self.xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]

            self.ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
            self.combcoordinates= np.meshgrid(self.xcoordinates,self.ycoordinates)

        except:
            raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')
        
        self.shape = self.xcoordinates.shape if self.ycoordinates is None else (len(self.xcoordinates),len(self.ycoordinates))

        # Showing results rate
        self.print_rate = args.print_rate
        self.show_rate = args.show_rate
        self.save_rate = args.save_rate
        self.surface_name=args.surface_name
        self.use_dataloader=args.use_dataloader
        
        self.epochs=args.epoch
        self.folders=args.folders
        self.batch_size= expargs['batch_size'] if args.batch_size==-1 else args.batch_size
        self.dataset= expargs['dataset'] if args.dataset=='' else args.dataset
        
        self.datasetparam= expargs['datasetparam']
        for k in args.datasetparam.keys():
            self.datasetparam[k]= args.datasetparam[k]
        
        self.loss= expargs['loss'] if args.loss=='' else args.loss
        self.lossparam= expargs['lossparam']
        for k in args.lossparam.keys():
            self.lossparam[k]= args.lossparam[k]


        # Load datasets
        print('Loading dataset: ',expargs['dataset'])
        if args.traindata:
            datajson='dataconfig_train.json'
        else:
            datajson='dataconfig_test.json'

        self.traindataset,self.train_loader, self.dmodule = loaddataset(datasetname=self.dataset,
                                        experimentparam=self.datasetparam,
                                        batch_size=self.batch_size,
                                        worker=args.worker,
                                        config_file=os.path.join(defaults_path,datajson))

        self.warp_var_mod = import_module( self.dmodule+'.dataset' )

        # Setup model
        print('Loading model: ',expargs['model'])
        if self.epochs!=-1:
            expargs['modelparam']['init']= os.path.join(folders['model_path'],'epoch{}model.t7'.format(self.epochs))
        else:
            expargs['modelparam']['init']= os.path.join(folders['model_path'],'lastmodel.t7')
        
        self.net, self.arch, self.mmodule = loadmodel(modelname=expargs['model'],
                                        experimentparams=expargs['modelparam'],
                                        config_file=os.path.join(defaults_path,'modelconfig.json'))

        self.net.to(self.device)
        self.w = copy.deepcopy(self.net.state_dict())

        netx=copy.deepcopy(self.net)
        if os.path.isfile(folders['plane_path_x']):
            init_type=folders['plane_path_x']
        else:
            init_type='xavier_uniform'
        init_params(netx,init_type)
        if not os.path.isfile(folders['plane_path_x']):
            normalize_directions_for_states(netx.state_dict(), self.net.state_dict(), args.xnorm, args.xignore)
        self.wx=create_target_direction(self.net,netx)
        if not os.path.isfile(folders['plane_path_x']):
            self.cx, self.cy= 1, 0
            self.savemodel(netx,folders['plane_path_x'])

        nety=copy.deepcopy(self.net)
        if os.path.isfile(folders['plane_path_y']):
            init_type=folders['plane_path_y']
        else:
            init_type='xavier_uniform'
        init_params(nety,init_type)
        if not os.path.isfile(folders['plane_path_y']):
            normalize_directions_for_states(nety.state_dict(), self.net.state_dict(), args.ynorm, args.yignore)
        self.wy=create_target_direction(self.net,nety)
        if not os.path.isfile(folders['plane_path_y']):
            self.cx, self.cy= 0, 1
            self.savemodel(nety,folders['plane_path_y'])

        self.cx, self.cy= 0, 0

        if self.use_parallel:
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        # Setup Loss criterion
        print('Selecting loss function: ',self.loss)
        self.criterion, self.losseval = selectloss(lossname=self.loss,
                                        parameter=self.lossparam,
                                        config_file=os.path.join(defaults_path,'loss_definition.json'))
        self.criterion.to(self.device)
        self.lossmat=np.zeros(shape=self.shape)
        
        # Others evaluation metrics
        print('Selecting metrics functions:')
        metrics_dict=get_metric_path(os.path.join(defaults_path,'metrics.json'))
        self.metrics = dict()
        self.metrics_eval = dict()
        self.metricsmat = dict()

        for key,value in metrics_dict.items():
            self.metrics[key],self.metrics_eval[key] = selectloss(lossname=value['metric'],
                                        parameter=value['param'],
                                        config_file=os.path.join(defaults_path,'loss_definition.json'))
            self.metrics[key].to(self.device)
            self.metricsmat[key]=np.zeros(shape=self.shape)
        
        self.computedmat=np.zeros(shape=self.shape)
        
        self.args=args

    def do_plot(self):
        k=-1
        total=len(self.xcoordinates)*len(self.ycoordinates)
        for cx in range(len(self.xcoordinates)):
            self.cx=cx
            for cy in range(len(self.ycoordinates)):
                k+=1
                self.cy=cy
                if os.path.isfile(self.folders['surface_path']+"/"+self.surface_name+".mat"):
                    surf=scipy.io.loadmat(self.folders['surface_path']+"/"+self.surface_name+".mat")
                    loadcomp=surf['computed']
                else:
                    loadcomp=self.computedmat  
                if (k % self.show_rate)==0 or k==(total-1):
                    self.visdom_plot()
                if loadcomp[cx,cy]==1:
                    self.computedmat[cx,cy]=1
                    self.lossmat[cx,cy]=surf['loss'][cx,cy]
                    for key,value in self.metrics_eval.items():
                        self.metricsmat[key][cx,cy]=surf[key][cx,cy]
                    continue

                coord=(self.combcoordinates[0][self.cx][self.cy] ,self.combcoordinates[1][self.cx][self.cy])
                #code for checking if exist
                set_states(self.net.module if self.use_parallel else self.net,self.w,self.wx,self.wy, coord)
                if self.use_dataloader:
                    self.validation_dl(cx,cy)
                else:
                    self.validation(cx,cy)
                if (k % self.save_rate)==0 or k==(total-1):
                    info = {'loss':self.lossmat}
                    
                    for key,value in self.metrics_eval.items():
                        info[key]=self.metricsmat[key]
                    
                    info['computed']=self.computedmat
                    scipy.io.savemat(self.folders['surface_path']+"/"+self.surface_name+".mat", info) 



    def validation(self,cx,cy): 
        ttime=time.time()  
        
        print('Computing {} {}:'.format(cx,cy), end=' ')
        lendata=len(self.traindataset)
        rperm=np.random.permutation(lendata)
        with torch.no_grad():    
            for i in range(lendata):
                sample=self.traindataset[rperm[i]]
                sample = self.warp_var_mod.warp_Variable(sample,self.device)
                for v in sample.keys():
                    if isinstance(sample[v],torch.Tensor):
                        sample[v].unsqueeze_(0)
                images=sample['image']
                

                outputs = self.net(images)
                kwarg=eval(self.losseval)
                loss=self.criterion(**kwarg)
                self.lossmat[cx,cy]+=loss.item()

                for key,value in self.metrics_eval.items():
                    kwarg=eval(self.metrics_eval[key])
                    metric=self.metrics[key](**kwarg)
                    self.metricsmat[key][cx,cy]+=metric.item()
                
                if (i+1)==self.batch_size:
                    break
        
        self.lossmat[cx,cy]/= i+1
        for key,value in self.metrics_eval.items():
            self.metricsmat[key][cx,cy]/=i+1

        self.computedmat[cx,cy]=1
        print(time.time()-ttime)
        
    def validation_dl(self,cx,cy): 
        ttime=time.time()  
        
        print('Computing {} {}:'.format(cx,cy), end=' ')
        with torch.no_grad():    
            for i,sample in enumerate(self.train_loader):
                sample = self.warp_var_mod.warp_Variable(sample,self.device)
                images=sample['image']
                

                outputs = self.net(images)
                kwarg=eval(self.losseval)
                loss=self.criterion(**kwarg)
                self.lossmat[cx,cy]+=loss.item()

                for key,value in self.metrics_eval.items():
                    kwarg=eval(self.metrics_eval[key])
                    metric=self.metrics[key](**kwarg)
                    self.metricsmat[key][cx,cy]+=metric.item()
                
                if (i+1)==self.batch_size:
                    break
        
        self.lossmat[cx,cy]/= i+1
        for key,value in self.metrics_eval.items():
            self.metricsmat[key][cx,cy]/=i+1

        self.computedmat[cx,cy]=1
        print(time.time()-ttime)

    def visdom_plot(self):
        if self.visdom==True:
            info = {'loss':self.lossmat}
            
            for key,value in self.metrics_eval.items():
                info[key]=self.metricsmat[key]
            for tag, value in info.items():
                if tag=='loss' or tag=='wce':
                    value=np.clip(value,0,5)
                self.vissurf.show('Landscape '+tag,'train', value)


    def loadmodel(self,net,modelpath):
        if os.path.isfile(modelpath):
            checkpoint = torch.load(modelpath,map_location='cpu')
            to_load= net.module if self.use_parallel else net
            to_load.load_state_dict(checkpoint['net'])
        else:
            raise Exception('Model not found')

    def savemodel(self,net,modelpath=''):
        if modelpath=='':
            print('Saving checkpoint  {} {}\n'.format(self.cx,self.cy))
            modelpath=os.path.join(self.folders['surface_path'],'model{}_{}.t7'.format(self.cx,self.cy))
        to_save= net.module if self.use_parallel else net
        state = {
                'arch':  self.arch,
                'net':   to_save.state_dict()
            }
        torch.save(state, modelpath)
