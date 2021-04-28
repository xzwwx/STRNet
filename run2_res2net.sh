#!/usr/bin/env bash

mainFolder="runs_log"
subFolder=$(date "+%Y%m%d-%H%M%S")


mkdir -p ${mainFolder}
mkdir -p ${mainFolder}/${subFolder}

CUDA_VISIBLE_DEVICES=3 python -u main_res2net.py 2>&1 | tee -a ${mainFolder}/${subFolder}/log.txt