#!/bin/bash

#SBATCH --partition=isg,multi-gpu,vlincs
#SBATCH --nodelist=iarpa018
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --job-name=V-gpt20
#SBATCH --time=256:00:00




source ~/venv/vllm/bin/activate

nvidia-smi

# https://huggingface.co/openai/gpt-oss-20b


# vllm serve openai/gpt-oss-20b --port 18443 #--tensor-parallel-size 1

# (Optional but recommended) Disable the cache to force a recompile
# export VLLM_DISABLE_COMPILE_CACHE=1

# VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 
vllm serve openai/gpt-oss-20b --max-model-len 16000 --tensor-parallel-size 1 --async-scheduling




