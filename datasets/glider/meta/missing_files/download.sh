#!/bin/bash
SAS="sv=2020-10-02&st=2022-02-14T13%3A13%3A03Z&se=2022-03-15T13%3A13%3A00Z&sr=c&sp=rl&sig=cFj6Y28%2B0mt0bCdTT55iar4GjdbwGQN0cfxHLWCHtJM%3D"
URL="https://n00244swesteurope.blob.core.windows.net/demo2000glider-kongsberg/Survey2018/Bulkdata/Hydrophone/Deployment_2"
DATAPATH=$URL?$SAS
echo $DATAPATH

azcopy list $DATAPATH --machine-readable

tmp_download_dir="/cluster/work/martimoa/tmp_download_dir/"
missing_files="/Users/martinmoan/Documents/Master/Year2/Thesis/TDT4900-Noise-in-the-Sea/code/datasets/glider/meta/missing_files/missing.txt"
azcopy copy $DATAPATH $tmp_download_dir --list-of-files $missing_files

