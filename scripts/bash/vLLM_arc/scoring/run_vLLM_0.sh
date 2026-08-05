#!/bin/bash
#SBATCH --job-name=rcn_vLLM_run_0
#SBATCH --account=sta_inf
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --nodelist=darkshadow-slurm-node-1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/scoring_%j.out
#SBATCH --error=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/scoring_%j_vLLM.err

set -euo pipefail

source ~/RecallNet/src/.venv/bin/activate
export PYTHONPYCACHEPREFIX=/tmp/pycache_$SLURM_JOB_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

export RECALLNET_VLLM_MAX_LEN=4096
export RECALLNET_VLLM_MAX_BATCHED_TOKENS=32768
export RECALLNET_VLLM_MAX_SEQS=512

cd ~/RecallNet/src/scripts

uv run vLLM_classify_2.py \
    --data-dir quasimodo_chunked \
    --dataset-prefix q \
    --predicates "has a" \
    --fp-num 0


