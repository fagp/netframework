#!/bin/bash

cd "$1"

mkdir "dataloaders"
mkdir "dataloaders/dataset"
mkdir "defaults"
mkdir "loss"
mkdir "models"
mkdir "models/arch"
mkdir "netutil"

git clone https://github.com/fagp/netframework.git
