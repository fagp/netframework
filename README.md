# bbsort
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

-include import to .arch.[archname] in src/models/loadmodel.py where archname is network module name

-model params inits must be defined using modelconfig.json file for default and --modelparam argument for experiment.

