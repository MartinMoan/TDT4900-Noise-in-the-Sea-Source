#!/bin/sh
conda init
conda deactivate
conda env remove --name TDT4900
conda env create --file ../environment.yml
conda activate TDT4900
pip install timm==0.4.5
# pip install pytorch-lightning
# Ensure that the directory /cluster/work/<username> exists. 