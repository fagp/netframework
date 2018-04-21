from netutil.NetFramework import *
import argparse
import json
import time
from utils.utils import Decoder

def main():
    # Common params
    parser = argparse.ArgumentParser(description='Net framework arguments description')
    parser.add_argument('--experiment', nargs='?', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--model', nargs='?', type=str, default='vgg16_2', help='Architecture to use')
    parser.add_argument('--modelparam', type=str, default='{}', help='Experiment model parameters')
    parser.add_argument('--dataset', nargs='?', type=str, default='blur', help='Dataset key specified in dataconfig_*.json')
    parser.add_argument('--datasetparam', type=str, default='{}', help='Experiment dataset parameters')
    parser.add_argument('--imsize', nargs='?', type=int, default=200, help='Image resize parameter')

    parser.add_argument('--visdom', action='store_true', help='If included shows visdom visulaization')
    parser.add_argument('--show_rate', nargs='?',type=int, default=4, help='Visdom show after num of iterations (used with --visdom)')
    parser.add_argument('--print_rate', nargs='?',type=int, default=4, help='Print after num of iterations')
    parser.add_argument('--save_rate', nargs='?',type=int, default=10, help='Save after num of iterations (if --save_rate=0 then no save is done during training)')

    parser.add_argument('--use_cuda', nargs='?',type=int, default=0, help='GPU device (if --use_cuda=-1 then CPU used)')
    parser.add_argument('--parallel', action='store_true', help='Use multiples GPU (used only if --use_cuda>-1)')
    parser.add_argument('--epochs', nargs='?', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, help='Minibatch size')
    parser.add_argument('--train_worker', nargs='?', type=int, default=1, help='Number of training workers')
    parser.add_argument('--test_worker', nargs='?', type=int, default=1, help='Number of testing workers')

    parser.add_argument('--optimizer', nargs='?', type=str, default='RMSprop', help='Optimizer to use')
    parser.add_argument('--optimizerparam', type=str, default='{}', help='Experiment optimizer parameters')
    parser.add_argument('--lrschedule', nargs='?', type=str, default='none', help='LR Schedule to use')
    parser.add_argument('--loss', nargs='?', type=str, default='ce', help='Loss function to use')
    parser.add_argument('--lossparam', type=str, default='{}', help='Loss function parameters')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    

    #Specific params################################################################################


    ################################################################################################

    #Parse arguments
    args = parser.parse_args()
    
    #For debug params################################################################################
    args.datasetparam="{'burstsize':'5'}"
    args.optimizerparam="{'lr':'0.00001'}"
    ################################################################################################


    args.lossparam=json.loads(args.lossparam.replace("'","\""),cls=Decoder)
    args.datasetparam=json.loads(args.datasetparam.replace("'","\""),cls=Decoder)
    args.modelparam=json.loads(args.modelparam.replace("'","\""),cls=Decoder)
    args.optimizerparam=json.loads(args.optimizerparam.replace("'","\""),cls=Decoder)

    # create outputs folders
    root='../out'
    experimentpath=(os.path.join(root,args.experiment))
    args.folders={ 'root_path':root, 'experiment_path':experimentpath, 'model_path':os.path.join(experimentpath,'model'), 'images_path':os.path.join(experimentpath,'images') }
    
    for folder, path in args.folders.items():
        if not os.path.isdir(path):
            os.mkdir(path)  

    sortingDL = NetFramework(args)
    start=time.time()
    sortingDL.do_train()
    print('Total Training Time {:.3f}'.format(time.time()-start))

if __name__ == '__main__':
    main()
