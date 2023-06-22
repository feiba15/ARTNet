# ARTNet

## Introduction

This is the code of the ARTNet model.

## Running

python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --use_env main.py --ema \
--combine_datasets=vidstg --combine_datasets_val=vidstg \
--output-dir=OUTPUT_DIR

