#!/bin/bash

#SBATCH --partition=isg,multi-gpu
#SBATCH --nodelist=delta
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --job-name=V-eE5-7B
#SBATCH --time=256:00:00




source ~/venv/vllm/bin/activate

nvidia-smi

# pn131276



#VLLM_DISABLE_COMPILE_CACHE=1 
vllm serve intfloat/e5-mistral-7b-instruct --port 18445 --tensor-parallel-size 1 --disable-log-requests --task=embed


# --task=embedding embed


