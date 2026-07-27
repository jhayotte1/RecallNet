#!/bin/bash
#SBATCH --job-name=recallnet_8b_test
#SBATCH --account=sta_inf
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodelist=arcadia-slurm-node-3

#SBATCH --output=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/slurm_%j.out
#SBATCH --error=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/slurm_%j.err

set -euo pipefail

source ~/.bashrc
conda activate recallnet

export OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 1000))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT

export RECALLNET_MODEL="llama3.1:8b"
export RECALLNET_BATCH_SIZE=10
export RECALLNET_MAX_CONCURRENCY=20
export RECALLNET_PROMPT="prompt_classify.txt"


OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=20 ollama serve > ~/RecallNet/ollama_$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
sleep 10

PREDS=("at location" "capable of")

cd /mnt/beegfs/projects/RecallNet/src/scripts
python3 LG_classify.py \
    --data-dir sample_data_1k_top_100k \
    --exp-name exp00 \
    --exp-desc "Test speed H100 with higher concurrency" \
    --predicates "${PREDS[@]}"

sleep 3
kill $OLLAMA_PID
