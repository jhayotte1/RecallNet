#!/bin/bash
#SBATCH --job-name=recallnet_8b_19
#SBATCH --gres=gpu:1
#SBATCH --time=23:50:00
#SBATCH --exclude=n51,n52,n53,n54,n55,n101,n102
#SBATCH --output=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.out
#SBATCH --error=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.err

source ~/.bashrc
conda activate recallnet

export OLLAMA_PORT=$((11434 + SLURM_JOB_ID % 1000))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT

export RECALLNET_MODEL="llama3.1:8b"
export RECALLNET_BATCH_SIZE=10
export RECALLNET_MAX_CONCURRENCY=10
export RECALLNET_PROMPT="prompt_classify.txt"


OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=10 ollama serve > ~/RecallNet/ollama_$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
sleep 10

PREDS=("has property_2")

cd /mnt/beegfs/projects/RecallNet/src/scripts
python3 LG_classify_2.py \
    --data-dir top_5M_by_predicate \
    --exp-name exp08 \
    --exp-desc "Final scoring, first pass" \
    --predicates "${PREDS[@]}"

sleep 3
kill $OLLAMA_PID
