#!/bin/bash

cmd="srun --mem 32G --time 12:0:0 -c5 --gres=gpu:1 --constraint volta -p dgx-spa,gpu,gpu-nvlink"
hparams="hyperparams/CRDNN-E.yaml"
py_script="train-dynbatch.py"

. path.sh
. kaldi-s5/utils/parse_options.sh

$cmd python $py_script $hparams
