#!/bin/sh
conda deactivate
conda env remove --name TDT4900
conda env create --file environment.yml
conda activate TDT4900
# Ensure that the directory /cluster/work/<username> exists. 