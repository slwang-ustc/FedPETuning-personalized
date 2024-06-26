#!/usr/bin/env bash

run_dir=$1
task=$2
clients_num=$3
alpha=$4

python ${run_dir}/code/FedPETuning/tools/glue_scripts/glue.py \
--data_dir ${run_dir}/data/glue/ \
--task ${task} \
--output_dir ${run_dir}/data \
--clients_num ${clients_num} \
--alpha ${alpha}
