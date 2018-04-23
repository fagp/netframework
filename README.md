# Pytorch NetFramework

Pytorch framework for quick implementation

## Dataset
1-Implementation of dataset.py in module [customdataset]

FILE:

	src/dataloaders/[customdataset]/dataset.py

IMPLEMENT:

	class cdataset(data.Dataset)

	def warp_Variable(sample,use_cuda=False,grad=True)

CONSTRAINT:

	cdataset.__getitem__(...) must return a dictionary with a key 'image'


2-Implementation of ctransforms.py in module [customdataset]

FILE:

	src/dataloaders/[customdataset]/ctransforms.py

IMPLEMENT:

	class ToTensor(object)

	problem specific transformation class

CONSTRAINT:

	Training transformations receive and return a dictionary with same keys as cdataset.__getitem__(...)


3-Default train/test paramenters

FILE:

	src/defaults/dataconfig_train.json

	src/defaults/dataconfig_test.json

DEFINE:

	use a json for cdataset.__init__(...) parameters initialization where keys are __init__ parameters names

CONSTRAINT:

	in dataconfig_[train/test].json must be defined "module" key where the value is the name of cdataset module (i.e. customdataset)

	in dataconfig_[train/test].json must be defined "transform_param" key where the value is a string with a sequence of transformations in ctransforms

NOTE:

	default behaivor could be overwritten for a specific experiment with --datasetparam argument, (i.e. --datasetparam="{'transform_param':'RandomCrop((100,100)),ToTensor()'}" ).


## Model

4-Implementation of arch/[model].py

FILE:

	src/models/arch/[model].py

IMPLEMENT:

	class [myarch](nn.Module):

CONSTRAINT:

	[myarch].__init__(...) return a nn.Module


5-Default model paramenters

FILE:

	src/defaults/modelconfig.json

DEFINE:

	use a json for [myarch].__init__(...) parameters initialization where keys are __init__ parameters names

CONSTRAINT:

	in modelconfig.json must be defined "module" key where the value is the name of [myarch] package (i.e. arch.model)

	in modelconfig.json must be defined "arch" key where the value is a string with architecture class name (i.e. myarch)

NOTE:

	default behaivor could be overwritten for a specific experiment with --modelparam argument, (i.e. --modelparam="{'num_class':'2'}" ).


## Metrics definitions

6- Implement custom metrics criterion

FILE:

	src/loss/lossfunc.py

IMPLEMENT:

	class [metric](nn.Module)

CONSTRAINT:

	[metric].__init__(...) return a nn.Module


7- Problem specific metrics definition

FILE:

	src/defaults/loss_definition.json

DEFINE:

	use a json for criterion module reference and call parameters dictionary specification

CONSTRAINT:

	in loss_definition.json must be defined a "criterion" key which is a reference to criterion class of step 6. Its [metric].__init__(...) parameters will be initialized with --lossparam argument for loss function and src/defaults/metrics.json for other metrics.

	in loss_definition.json must be defined a "criterionparam" key where are specified criterion call parameters [metric].forward(criterionparam). Variables names used must match with outputs for network forward and sample[...] for custom dataset keys


8- Metrics to be used in network analysis

FILE:

	src/defaults/metrics.json

DEFINE:

	use a json to specify the list of metrics to be computed

CONSTRAINT:

	in metrics.json must be defined a "metric" key which is one of the criterions defined in loss_definition.json

	in metrics.json must be defined a "param" key with initialization [metric].__init__(param)

## Custom Visualization

9- Implement custom behavior of NetFramework using python inherit

FILE:

	src/netutil/[customnet].py

IMPLEMENT:

	class [CustomNet](NetFramework):
		def __init__(self,args):
		        NetFramework.__init__(self,args)

		def valid_visualization(self,current_epoch,index=0,save=False)
			#do visualization

CONSTRAINT:

	[CustomNet] must inherit from NetFramework

	valid_visualization function must use NetFramework attributes


