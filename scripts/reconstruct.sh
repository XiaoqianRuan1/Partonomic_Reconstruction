#!/bin/bash
set -e
#CUDA_VISIBLE_DEVICES=$cuda python src/reconstruct.py --model $model --input $input
#CUDA_VISIBLE_DEVICES=$cuda python src/qual_results.py --category $category --model $model --input $input --output $output
CUDA_VISIBLE_DEVICES=$cuda python src/cub_result.py --model $model --input $input --output $output