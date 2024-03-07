#!/bin/bash

#Please modify the following roots to yours.
dataset_root=/media/userdisk0/zychen/Datasets/Polyp
model_root=/media/userdisk0/zychen/RandomTIMDA/POLYP/models/
path_save_log=/media/userdisk0/zychen/RandomTIMDA/POLYP/logs/

#Dataset [BKAI, CVC-ClinicDB, ETIS-LaribPolypDB, Kvasir-SEG]
Source=BKAI

#Optimizer
optimizer=Adam
lr=0.01

#Hyperparameters
memory_size=40
neighbor=16
prompt_alpha=0.01
warm_n=5

#Command
cd POLYP
CUDA_VISIBLE_DEVICES=0 python vptta.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--memory_size $memory_size --neighbor $neighbor --prompt_alpha $prompt_alpha --warm_n $warm_n