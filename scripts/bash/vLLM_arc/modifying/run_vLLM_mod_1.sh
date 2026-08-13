#!/bin/bash
#SBATCH --job-name=rcn_vLLM_run
#SBATCH --account=sta_inf
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --nodelist=darkshadow-slurm-node-2
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=45G
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
export RECALLNET_PROMPT="prompt_modify.txt"

cd ~/RecallNet/src/scripts

uv run vLLM_classify_2.py \
    --data-dir quasimodo_chunked \
    --dataset-prefix q \
    --predicates "at location" "causes" "created by" "defined as" "desires" "has first subevent"

