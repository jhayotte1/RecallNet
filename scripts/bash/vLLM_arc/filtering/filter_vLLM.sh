#!/bin/bash
#SBATCH --job-name=rcn_filt_vLLM
#SBATCH --account=sta_inf
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --nodelist=darkshadow-slurm-node-2
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --output=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/scoring_%j.out
#SBATCH --error=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/scoring_%j_vLLM.err

set -euo pipefail

source ~/RecallNet/src/.venv/bin/activate
export PYTHONPYCACHEPREFIX=/tmp/pycache_$SLURM_JOB_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

export RECALLNET_VLLM_MODEL="/mnt/ssd/recallnet/models/Meta-Llama-3.3-70B-Instruct"
export RECALLNET_MODEL_LIGHT="llama3.3:70b-fp8"
export RECALLNET_VLLM_MAX_LEN=4096

cd ~/RecallNet/src/scripts

uv run vLLM_filtering.py \
    --dataset-prefix q \
    --predicates "causes"

