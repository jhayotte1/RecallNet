#!/bin/bash
#SBATCH --job-name=recallnet_8b_arc
#SBATCH --account=sta_inf
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodelist=arcadia-slurm-node-3

#SBATCH --output=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/slurm_%j.out
#SBATCH --error=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/slurm_%j.err

set -euo pipefail

source ~/.bashrc
cd ~/RecallNet/src
source .venv/bin/activate


export OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 1000))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT

export RECALLNET_MODEL="llama3.1:8b"
export RECALLNET_BATCH_SIZE=10
export RECALLNET_MAX_CONCURRENCY=10
export RECALLNET_PROMPT="prompt_classify.txt"

export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=10
export OLLAMA_CONTEXT_LENGTH=8192

ollama serve > ~/RecallNet/ollama_$SLURM_JOB_ID.log 2>&1 &

OLLAMA_PID=$!
sleep 10

PREDS=("has property_1")

cd scripts

python3 LG_classify_2.py \
    --data-dir top_5M_by_predicate \
    --exp-name exp00 \
    --exp-desc "Final scoring" \
    --predicates "${PREDS[@]}"

sleep 3
kill $OLLAMA_PID
