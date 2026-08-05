#!/bin/bash
#SBATCH --job-name=rcn_vLLM_run_11
#SBATCH --account=sta_inf
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --nodelist=darkshadow-slurm-node-1
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

export RECALLNET_VLLM_MAX_LEN=4096

cd ~/RecallNet/src/scripts

uv run vLLM_classify_2.py \
    --data-dir top_5M_by_predicate \
    --dataset-prefix q \
    --predicates "at location" "capable of" "causes" "created by" "defined as" \
    --fp-num 11


