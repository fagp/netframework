#!/bin/bash

mkdir ../dataloaders
mkdir ../dataloaders/customdataset
touch ../dataloaders/__init__.py
touch ../dataloaders/customdataset/__init__.py
touch ../dataloaders/customdataset/dataset.py
touch ../dataloaders/customdataset/ctransforms.py

mkdir ../defaults
touch ../defaults/dataconfig_test.json
touch ../defaults/dataconfig_train.json
touch ../defaults/loss_definition.json
touch ../defaults/metrics.json
touch ../defaults/modelconfig.json

mkdir ../loss
touch ../loss/__init__.py
touch ../loss/lossfunc.py

mkdir ../models
mkdir ../models/arch
touch ../models/__init__.py
touch ../models/arch/__init__.py

mkdir ../netutil
touch ../netutil/__init__.py
touch ../netutil/customnet.py
