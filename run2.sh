#!/usr/bin/env bash

mainFolder="runs_log"
subFolder=$(date "+%Y%m%d-%H%M%S")


mkdir -p ${mainFolder}
mkdir -p ${mainFolder}/${subFolder}

CUDA_VISIBLE_DEVICES=0,1 python -u main22.py 2>&1 | tee -a ${mainFolder}/${subFolder}/log.txt
