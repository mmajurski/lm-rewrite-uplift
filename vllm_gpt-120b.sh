#!/bin/bash

#SBATCH --partition=isg,multi-gpu
#SBATCH --nodelist=sierra
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --job-name=V-gpt120
#SBATCH --time=256:00:00




source ~/venv/vllm/bin/activate

nvidia-smi

# https://huggingface.co/openai/gpt-oss-20b


# vllm serve openai/gpt-oss-120b --port 18444 #--tensor-parallel-size 1

#VLLM_DISABLE_COMPILE_CACHE=1 
vllm serve openai/gpt-oss-120b --port 18443 --served-model-name gpt-oss-120b --max-model-len 72000 --tensor-parallel-size 4 --async-scheduling
#--task=embedding


