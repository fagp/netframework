# Pytorch NetFramework
For dataset specifications:

src/defaults/dataconfig_train.json
src/defaults/dataconfig_test.json
src/dataloaders/customdataset/ctransforms.py
src/dataloaders/customdataset/dataset.py

-implement customs ctransforms.py and dataset.py

-dataset params inits must be defined using dataconfig_*.json files for default and --datasetparam  argument for experiment.

-Transformation must be defined with transform_param key in dataconfig_*.json for default and --datasetparam argument for experiment. 


For model specifications:

src/defaults/modelconfig.json
src/models/arch/*.py
src/models/loadmodel.py

-implement customs architectures in src/models/arch/*.py

-model params inits must be defined using modelconfig.json file for default and --modelparam argument for experiment.


For loss specifications:

src/defaults/metrics.json
src/loss/selectloss.py
src/loss/lossfunc.py

-implement customs loss in src/loss/lossfunc.py

-include loss function in src/loss/selectloss.py

-list others metrics src/defaults/metrics.json


